from dataclasses import dataclass


from typing import Dict, List, Optional, Tuple





import torch


from torch import Tensor








@dataclass(frozen=True)


class MeshBatch:


    position: Tensor


    normal: Tensor


    curvature: Tensor


    wall_distance: Tensor


    edge_index: Tensor


    edge_length: Tensor


    batch: Tensor


    volume_cells: Tensor


    surface_cells: Tensor


    patient_ids: Tuple[str, ...]





    def to(self, device: torch.device) -> "MeshBatch":


        return MeshBatch(


            position=self.position.to(device),


            normal=self.normal.to(device),


            curvature=self.curvature.to(device),


            wall_distance=self.wall_distance.to(device),


            edge_index=self.edge_index.to(device),


            edge_length=self.edge_length.to(device),


            batch=self.batch.to(device),


            volume_cells=self.volume_cells.to(device),


            surface_cells=self.surface_cells.to(device),


            patient_ids=self.patient_ids,


        )








@dataclass


class FieldBattery:


    pressure: Tensor


    velocity: Tensor


    wss: Tensor


    tawss: Tensor


    osi: Tensor


    ecap: Tensor


    ilt: Tensor


    pws: Tensor


    rpi: Tensor


    quantiles: Optional[Dict[str, Tensor]] = None





    def tensors(self) -> Dict[str, Tensor]:


        return {


            "pressure": self.pressure,


            "velocity": self.velocity,


            "wss": self.wss,


            "tawss": self.tawss,


            "osi": self.osi,


            "ecap": self.ecap,


            "ilt": self.ilt,


            "pws": self.pws,


            "rpi": self.rpi,


        }





    def detach(self) -> "FieldBattery":


        values = {name: value.detach() for name, value in self.tensors().items()}


        quantiles = None


        if self.quantiles is not None:


            quantiles = {name: value.detach() for name, value in self.quantiles.items()}


        return FieldBattery(**values, quantiles=quantiles)








@dataclass(frozen=True)


class GeometryBounds:


    diameter: Tuple[float, float]


    curvature: Tuple[float, float]


    tortuosity: Tuple[float, float]


    volume: Tuple[float, float]


    ilt_fraction: Tuple[float, float]








@dataclass(frozen=True)


class OptimizerState:


    epoch: int


    global_step: int


    best_validation_loss: float


    seed: int


    wall_slack: float








@dataclass(frozen=True)


class CohortRecord:


    patient_id: str


    site: str


    region: str


    split: str


    mesh_path: str


    field_path: str


    diameter_mm: float


    sex: int


    age: float


    event: Optional[int]


    event_time_days: Optional[float]


    scanner_vendor: str








@dataclass(frozen=True)


class EvaluationSummary:


    field_metrics: Dict[str, Dict[str, float]]


    clinical_metrics: Dict[str, float]


    site_metrics: Dict[str, Dict[str, float]]


    subgroup_metrics: Dict[str, Dict[str, float]]


    abstention_metrics: Dict[str, float]


