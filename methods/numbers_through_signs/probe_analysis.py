from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RidgeFit:
    train_mean: np.ndarray
    target_mean: float
    train_centered: np.ndarray
    dual_weights: np.ndarray

    def predict(self, test: np.ndarray) -> np.ndarray:
        centered = test.astype(np.float64) - self.train_mean
        return centered @ self.train_centered.T @ self.dual_weights + self.target_mean


def fit_dual_ridge(train: np.ndarray, target: np.ndarray, alpha: float) -> RidgeFit:
    x = train.astype(np.float64)
    y = target.astype(np.float64)
    train_mean = x.mean(axis=0)
    target_mean = float(y.mean())
    centered = x - train_mean
    centered_target = y - target_mean
    gram = centered @ centered.T
    feature_scale = float(np.trace(gram) / max(len(x), 1))
    regularizer = max(feature_scale * alpha, np.finfo(np.float64).eps)
    dual = np.linalg.solve(gram + regularizer * np.eye(len(x), dtype=np.float64), centered_target)
    return RidgeFit(train_mean, target_mean, centered, dual)


def _aggregate_predictions(metadata: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    frame = metadata[["template_family", "value"]].copy()
    frame["prediction"] = predictions
    return frame.groupby(["template_family", "value"], as_index=False).prediction.mean()


def regression_metrics(metadata: pd.DataFrame, predictions: np.ndarray) -> dict[str, float]:
    aggregate = _aggregate_predictions(metadata, predictions)
    expected = aggregate.value.to_numpy(dtype=np.float64)
    predicted = aggregate.prediction.to_numpy(dtype=np.float64)
    errors = predicted - expected
    return {"mae": float(np.abs(errors).mean())}


def select_alpha(train: np.ndarray, metadata: pd.DataFrame, candidates: tuple[float, ...]) -> float:
    families = sorted(metadata.template_family.unique())
    scores: dict[float, list[float]] = {alpha: [] for alpha in candidates}
    values = metadata.value.to_numpy(dtype=np.float64)
    for held_out in families:
        validation = metadata.template_family.eq(held_out).to_numpy()
        fitting = ~validation
        if fitting.sum() < 2 or validation.sum() == 0:
            continue
        for alpha in candidates:
            model = fit_dual_ridge(train[fitting], values[fitting], alpha)
            metrics = regression_metrics(metadata.loc[validation], model.predict(train[validation]))
            scores[alpha].append(metrics["mae"])
    usable = {alpha: float(np.mean(values)) for alpha, values in scores.items() if values}
    if not usable:
        return candidates[len(candidates) // 2]
    return min(usable, key=lambda alpha: (usable[alpha], alpha))
