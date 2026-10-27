from dataclasses import dataclass


from typing import Dict, Iterable, List, Mapping, Sequence, Tuple





import numpy as np


from scipy import stats


from sklearn.linear_model import LogisticRegression


from sklearn.metrics import roc_auc_score, roc_curve








@dataclass(frozen=True)


class LogisticBiomarker:


    coefficients: np.ndarray


    intercept: float


    feature_mean: np.ndarray


    feature_scale: np.ndarray





    def predict(self, features: np.ndarray) -> np.ndarray:


        standardized = (features - self.feature_mean) / self.feature_scale


        logits = standardized @ self.coefficients + self.intercept


        return 1.0 / (1.0 + np.exp(-logits))








def fit_firewalled_biomarker(features: np.ndarray, outcomes: np.ndarray, penalty: float = 1.0) -> LogisticBiomarker:


    mean = features.mean(axis=0)


    scale = features.std(axis=0, ddof=1)


    scale = np.where(scale > 0, scale, 1.0)


    standardized = (features - mean) / scale


    model = LogisticRegression(C=1.0 / penalty, solver="lbfgs", max_iter=10000)


    model.fit(standardized, outcomes)


    return LogisticBiomarker(model.coef_[0].copy(), float(model.intercept_[0]), mean, scale)








def auc_interval(outcomes: np.ndarray, scores: np.ndarray, repetitions: int = 10000, seed: int = 1701) -> Tuple[float, float, float]:


    point = float(roc_auc_score(outcomes, scores))


    generator = np.random.default_rng(seed)


    values: List[float] = []


    for _ in range(repetitions):


        indices = generator.integers(0, len(outcomes), len(outcomes))


        if np.unique(outcomes[indices]).size < 2:


            continue


        values.append(float(roc_auc_score(outcomes[indices], scores[indices])))


    return point, float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))








def net_reclassification_improvement(outcomes: np.ndarray, baseline: np.ndarray, updated: np.ndarray, thresholds: Sequence[float]) -> float:


    baseline_class = np.digitize(baseline, thresholds)


    updated_class = np.digitize(updated, thresholds)


    events = outcomes == 1


    controls = outcomes == 0


    event_up = np.mean(updated_class[events] > baseline_class[events])


    event_down = np.mean(updated_class[events] < baseline_class[events])


    control_down = np.mean(updated_class[controls] < baseline_class[controls])


    control_up = np.mean(updated_class[controls] > baseline_class[controls])


    return float(event_up - event_down + control_down - control_up)








def optimal_threshold(outcomes: np.ndarray, scores: np.ndarray) -> float:


    false_positive, true_positive, thresholds = roc_curve(outcomes, scores)


    return float(thresholds[np.argmax(true_positive - false_positive)])








def sensitivity_specificity(outcomes: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, float]:


    predicted = scores >= threshold


    positive = outcomes == 1


    negative = ~positive


    sensitivity = np.mean(predicted[positive])


    specificity = np.mean(~predicted[negative])


    ppv = np.sum(predicted & positive) / max(np.sum(predicted), 1)


    npv = np.sum(~predicted & negative) / max(np.sum(~predicted), 1)


    return {"sensitivity": float(sensitivity), "specificity": float(specificity), "ppv": float(ppv), "npv": float(npv)}








def holm_bonferroni(p_values: Mapping[str, float], alpha: float = 0.05) -> Dict[str, bool]:


    ordered = sorted(p_values.items(), key=lambda item: item[1])


    decisions: Dict[str, bool] = {name: False for name in p_values}


    for rank, (name, value) in enumerate(ordered):


        threshold = alpha / (len(ordered) - rank)


        if value <= threshold:


            decisions[name] = True


        else:


            break


    return decisions








def subgroup_auc(outcomes: np.ndarray, scores: np.ndarray, groups: np.ndarray) -> Dict[str, float]:


    result: Dict[str, float] = {}


    for group in np.unique(groups):


        selected = groups == group


        if np.unique(outcomes[selected]).size < 2:


            result[str(group)] = float("nan")


        else:


            result[str(group)] = float(roc_auc_score(outcomes[selected], scores[selected]))


    return result








def calibration_slope(outcomes: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:


    clipped = np.clip(scores, 1e-6, 1.0 - 1e-6)


    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=10000)


    model.fit(logits, outcomes)


    return float(model.intercept_[0]), float(model.coef_[0, 0])








def decision_curve(outcomes: np.ndarray, scores: np.ndarray, thresholds: Sequence[float]) -> Dict[float, float]:


    sample_count = len(outcomes)


    result: Dict[float, float] = {}


    for threshold in thresholds:


        predicted = scores >= threshold


        true_positive = np.sum(predicted & (outcomes == 1))


        false_positive = np.sum(predicted & (outcomes == 0))


        result[float(threshold)] = float(true_positive / sample_count - false_positive / sample_count * threshold / (1.0 - threshold))


    return result


