import csv


from pathlib import Path


from typing import Dict, Iterator, List, Mapping, Sequence, Tuple





import h5py


import numpy as np


import torch


from torch import Tensor


from torch.utils.data import Dataset, Sampler





from chaw_op.contracts import CohortRecord, FieldBattery, MeshBatch








class AAAMeshDataset(Dataset[Tuple[MeshBatch, FieldBattery, CohortRecord]]):


    def __init__(self, root: Path, manifest: Path, split: str) -> None:


        self.root = root


        self.records = self._read_manifest(manifest, split)





    def __len__(self) -> int:


        return len(self.records)





    def __getitem__(self, index: int) -> Tuple[MeshBatch, FieldBattery, CohortRecord]:


        record = self.records[index]


        mesh = self._read_mesh(self.root / record.mesh_path, record.patient_id)


        fields = self._read_fields(self.root / record.field_path)


        return mesh, fields, record





    def _read_manifest(self, path: Path, split: str) -> List[CohortRecord]:


        records: List[CohortRecord] = []


        with path.open("r", encoding="utf-8", newline="") as handle:


            reader = csv.DictReader(handle)


            for row in reader:


                if row["split"] != split:


                    continue


                records.append(


                    CohortRecord(


                        patient_id=row["patient_id"],


                        site=row["site"],


                        region=row["region"],


                        split=row["split"],


                        mesh_path=row["mesh_path"],


                        field_path=row["field_path"],


                        diameter_mm=float(row["diameter_mm"]),


                        sex=int(row["sex"]),


                        age=float(row["age"]),


                        event=int(row["event"]) if row.get("event", "") else None,


                        event_time_days=float(row["event_time_days"]) if row.get("event_time_days", "") else None,


                        scanner_vendor=row["scanner_vendor"],


                    )


                )


        return records





    def _read_mesh(self, path: Path, patient_id: str) -> MeshBatch:


        with h5py.File(path, "r") as handle:


            position = torch.from_numpy(np.asarray(handle["position"], dtype=np.float32))


            normal = torch.from_numpy(np.asarray(handle["normal"], dtype=np.float32))


            curvature = torch.from_numpy(np.asarray(handle["curvature"], dtype=np.float32))


            wall_distance = torch.from_numpy(np.asarray(handle["wall_distance"], dtype=np.float32))


            edge_index = torch.from_numpy(np.asarray(handle["edge_index"], dtype=np.int64))


            volume_cells = torch.from_numpy(np.asarray(handle["volume_cells"], dtype=np.int64))


            surface_cells = torch.from_numpy(np.asarray(handle["surface_cells"], dtype=np.int64))


        edge_length = torch.linalg.vector_norm(position[edge_index[1]] - position[edge_index[0]], dim=-1)


        return MeshBatch(


            position=position,


            normal=normal,


            curvature=curvature,


            wall_distance=wall_distance,


            edge_index=edge_index,


            edge_length=edge_length,


            batch=torch.zeros(position.shape[0], dtype=torch.long),


            volume_cells=volume_cells,


            surface_cells=surface_cells,


            patient_ids=(patient_id,),


        )





    def _read_fields(self, path: Path) -> FieldBattery:


        with h5py.File(path, "r") as handle:


            values = {


                name: torch.from_numpy(np.asarray(handle[name], dtype=np.float32))


                for name in ("pressure", "velocity", "wss", "tawss", "osi", "ecap", "ilt", "pws", "rpi")


            }


        return FieldBattery(**values)








class SiteBalancedSampler(Sampler[int]):


    def __init__(self, records: Sequence[CohortRecord], seed: int, samples: int | None = None) -> None:


        self.records = records


        self.seed = seed


        self.samples = samples or len(records)


        self.site_indices: Dict[str, List[int]] = {}


        for index, record in enumerate(records):


            self.site_indices.setdefault(record.site, []).append(index)





    def __len__(self) -> int:


        return self.samples





    def __iter__(self) -> Iterator[int]:


        generator = torch.Generator().manual_seed(self.seed)


        sites = sorted(self.site_indices)


        for draw in range(self.samples):


            site = sites[draw % len(sites)]


            indices = self.site_indices[site]


            selection = int(torch.randint(len(indices), (1,), generator=generator).item())


            yield indices[selection]








def collate_meshes(items: Sequence[Tuple[MeshBatch, FieldBattery, CohortRecord]]) -> Tuple[MeshBatch, FieldBattery, List[CohortRecord]]:


    meshes, fields, records = zip(*items)


    offsets: List[int] = []


    total = 0


    for mesh in meshes:


        offsets.append(total)


        total += mesh.position.shape[0]


    edge_index = torch.cat([mesh.edge_index + offset for mesh, offset in zip(meshes, offsets)], dim=1)


    volume_cells = torch.cat([mesh.volume_cells + offset for mesh, offset in zip(meshes, offsets)], dim=0)


    surface_cells = torch.cat([mesh.surface_cells + offset for mesh, offset in zip(meshes, offsets)], dim=0)


    batch = torch.cat([torch.full((mesh.position.shape[0],), index, dtype=torch.long) for index, mesh in enumerate(meshes)])


    merged_mesh = MeshBatch(


        position=torch.cat([mesh.position for mesh in meshes]),


        normal=torch.cat([mesh.normal for mesh in meshes]),


        curvature=torch.cat([mesh.curvature for mesh in meshes]),


        wall_distance=torch.cat([mesh.wall_distance for mesh in meshes]),


        edge_index=edge_index,


        edge_length=torch.cat([mesh.edge_length for mesh in meshes]),


        batch=batch,


        volume_cells=volume_cells,


        surface_cells=surface_cells,


        patient_ids=tuple(record.patient_id for record in records),


    )


    names = fields[0].tensors().keys()


    merged_fields = FieldBattery(**{name: torch.cat([field.tensors()[name] for field in fields]) for name in names})


    return merged_mesh, merged_fields, list(records)


