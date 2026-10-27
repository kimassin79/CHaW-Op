from typing import Dict, Mapping, Sequence, Tuple





import numpy as np


import torch


from torch import Tensor





from chaw_op.contracts import FieldBattery








def coefficient_of_determination(prediction: Tensor, target: Tensor) -> float:


    prediction = prediction.detach().double().flatten()


    target = target.detach().double().flatten()


    residual = (target - prediction).square().sum()


    total = (target - target.mean()).square().sum()


    return float((1.0 - residual / total.clamp_min(1e-12)).cpu())








def normalized_mae(prediction: Tensor, target: Tensor) -> float:


    absolute = (prediction.detach() - target.detach()).abs().mean()


    scale = target.detach().abs().amax() - target.detach().abs().amin()


    return float((absolute / scale.clamp_min(1e-12)).cpu())








def relative_l2(prediction: Tensor, target: Tensor) -> float:


    error = torch.linalg.vector_norm((prediction.detach() - target.detach()).flatten())


    scale = torch.linalg.vector_norm(target.detach().flatten())


    return float((error / scale.clamp_min(1e-12)).cpu())








def cosine_alignment(prediction: Tensor, target: Tensor) -> float:


    if prediction.shape[-1] == 1:


        return float("nan")


    values = torch.nn.functional.cosine_similarity(prediction.detach(), target.detach(), dim=-1)


    return float(values.mean().cpu())








def evaluate_fields(prediction: FieldBattery, target: FieldBattery) -> Dict[str, Dict[str, float]]:


    result: Dict[str, Dict[str, float]] = {}


    for name, values in prediction.tensors().items():


        reference = target.tensors()[name]


        result[name] = {


            "r2": coefficient_of_determination(values, reference),


            "nmae": normalized_mae(values, reference),


            "relative_l2": relative_l2(values, reference),


            "cosine": cosine_alignment(values, reference),


        }


    return result








def aggregate_seed_metrics(metrics: Sequence[Mapping[str, Mapping[str, float]]]) -> Dict[str, Dict[str, float]]:


    result: Dict[str, Dict[str, float]] = {}


    for field in metrics[0]:


        result[field] = {}


        for metric in metrics[0][field]:


            values = np.asarray([run[field][metric] for run in metrics], dtype=np.float64)


            result[field][f"{metric}_mean"] = float(np.nanmean(values))


            result[field][f"{metric}_sd"] = float(np.nanstd(values, ddof=1))


    return result








def bootstrap_interval(values: Sequence[float], confidence: float = 0.95, repetitions: int = 10000, seed: int = 1701) -> Tuple[float, float]:


    array = np.asarray(values, dtype=np.float64)


    generator = np.random.default_rng(seed)


    samples = generator.choice(array, size=(repetitions, len(array)), replace=True).mean(axis=1)


    tail = (1.0 - confidence) / 2.0


    return float(np.quantile(samples, tail)), float(np.quantile(samples, 1.0 - tail))








def spatial_coverage(lower: Tensor, upper: Tensor, target: Tensor) -> float:


    covered = (target >= lower) & (target <= upper)


    return float(covered.float().mean().cpu())


