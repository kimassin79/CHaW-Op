from dataclasses import dataclass


from typing import Dict, Iterable, List, Mapping, Sequence, Tuple





import numpy as np


import torch


from torch import Tensor





from chaw_op.contracts import FieldBattery, GeometryBounds








@dataclass(frozen=True)


class ConformalState:


    alpha: float


    field_quantiles: Dict[str, float]


    geometry_threshold: float


    calibration_count: int








class FunctionalConformalCalibrator:


    def __init__(self, alpha: float = 0.1) -> None:


        self.alpha = alpha


        self.state: ConformalState | None = None





    def fit(self, predictions: Sequence[FieldBattery], targets: Sequence[FieldBattery], geometry_scores: Sequence[float]) -> ConformalState:


        if len(predictions) != len(targets):


            raise ValueError("prediction and target counts differ")


        scores: Dict[str, List[float]] = {name: [] for name in predictions[0].tensors()}


        for prediction, target in zip(predictions, targets):


            for name, values in prediction.tensors().items():


                reference = target.tensors()[name]


                scale = reference.detach().abs().median().clamp_min(1e-6)


                error = ((values.detach() - reference.detach()).abs() / scale).flatten().max()


                scores[name].append(float(error.cpu()))


        quantiles = {name: self._finite_sample_quantile(values) for name, values in scores.items()}


        geometry_threshold = self._finite_sample_quantile(list(geometry_scores))


        self.state = ConformalState(self.alpha, quantiles, geometry_threshold, len(predictions))


        return self.state





    def intervals(self, prediction: FieldBattery) -> Dict[str, Tuple[Tensor, Tensor]]:


        if self.state is None:


            raise RuntimeError("calibrator is not fitted")


        intervals: Dict[str, Tuple[Tensor, Tensor]] = {}


        for name, values in prediction.tensors().items():


            radius = self.state.field_quantiles[name] * values.detach().abs().median().clamp_min(1e-6)


            intervals[name] = values - radius, values + radius


        return intervals





    def abstains(self, geometry_score: float) -> bool:


        if self.state is None:


            raise RuntimeError("calibrator is not fitted")


        return geometry_score > self.state.geometry_threshold





    def _finite_sample_quantile(self, scores: Sequence[float]) -> float:


        values = np.asarray(scores, dtype=np.float64)


        rank = int(np.ceil((len(values) + 1) * (1.0 - self.alpha)))


        rank = min(max(rank, 1), len(values))


        return float(np.partition(values, rank - 1)[rank - 1])








class GeometryNonconformity:


    def __init__(self, center: Tensor, precision: Tensor) -> None:


        self.center = center


        self.precision = precision





    @classmethod


    def fit(cls, descriptors: Tensor, regularization: float = 1e-4) -> "GeometryNonconformity":


        center = descriptors.mean(dim=0)


        centered = descriptors - center


        covariance = centered.t().matmul(centered) / max(descriptors.shape[0] - 1, 1)


        covariance = covariance + torch.eye(covariance.shape[0], device=covariance.device) * regularization


        precision = torch.linalg.inv(covariance)


        return cls(center, precision)





    def score(self, descriptor: Tensor) -> Tensor:


        difference = descriptor - self.center


        return torch.sqrt(torch.einsum("...i,ij,...j->...", difference, self.precision, difference).clamp_min(0.0))








def geometry_descriptor(position: Tensor, curvature: Tensor, wall_distance: Tensor) -> Tensor:


    extent = position.max(dim=0).values - position.min(dim=0).values


    covariance = torch.cov(position.t())


    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)


    values = torch.cat(


        [


            extent,


            eigenvalues,


            curvature.mean(dim=0),


            curvature.std(dim=0),


            wall_distance.mean(dim=0),


            wall_distance.std(dim=0),


        ]


    )


    return values


