import hashlib


from dataclasses import dataclass


from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple





import numpy as np





from chaw_op.contracts import CohortRecord








@dataclass(frozen=True)


class SplitAudit:


    counts: Dict[str, int]


    patient_overlap: Dict[str, int]


    site_membership: Dict[str, Tuple[str, ...]]


    event_rates: Dict[str, float]








def audit_splits(records: Sequence[CohortRecord]) -> SplitAudit:


    groups: Dict[str, List[CohortRecord]] = {}


    for record in records:


        groups.setdefault(record.split, []).append(record)


    patient_sets = {name: {record.patient_id for record in values} for name, values in groups.items()}


    overlap: Dict[str, int] = {}


    names = sorted(groups)


    for left_index, left in enumerate(names):


        for right in names[left_index + 1 :]:


            overlap[f"{left}:{right}"] = len(patient_sets[left] & patient_sets[right])


    counts = {name: len(values) for name, values in groups.items()}


    sites = {name: tuple(sorted({record.site for record in values})) for name, values in groups.items()}


    event_rates: Dict[str, float] = {}


    for name, values in groups.items():


        labels = [record.event for record in values if record.event is not None]


        event_rates[name] = float(np.mean(labels)) if labels else float("nan")


    return SplitAudit(counts, overlap, sites, event_rates)








def geometry_hash(vertices: np.ndarray, decimals: int = 4) -> str:


    centered = vertices - vertices.mean(axis=0, keepdims=True)


    scale = np.linalg.norm(centered, axis=1).max()


    normalized = centered / max(scale, 1e-12)


    ordered = normalized[np.lexsort((normalized[:, 2], normalized[:, 1], normalized[:, 0]))]


    return hashlib.sha256(np.round(ordered, decimals).tobytes()).hexdigest()








def verify_disjoint_hashes(pretraining: Iterable[str], evaluation: Iterable[str]) -> Set[str]:


    return set(pretraining) & set(evaluation)








def patient_level_partition(patient_ids: Sequence[str], fractions: Tuple[float, float, float], seed: int) -> Dict[str, str]:


    if not np.isclose(sum(fractions), 1.0):


        raise ValueError("fractions must sum to one")


    unique = np.asarray(sorted(set(patient_ids)))


    generator = np.random.default_rng(seed)


    generator.shuffle(unique)


    train_end = int(len(unique) * fractions[0])


    validation_end = train_end + int(len(unique) * fractions[1])


    mapping: Dict[str, str] = {}


    for patient in unique[:train_end]:


        mapping[str(patient)] = "train"


    for patient in unique[train_end:validation_end]:


        mapping[str(patient)] = "validation"


    for patient in unique[validation_end:]:


        mapping[str(patient)] = "test"


    return mapping


