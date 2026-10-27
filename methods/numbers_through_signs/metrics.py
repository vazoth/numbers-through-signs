from __future__ import annotations
from collections.abc import Sequence
import numpy as np


def class_logit(logits: np.ndarray, token_ids: Sequence[int]) -> np.ndarray:
    selected = logits[..., list(token_ids)]
    maximum = np.max(selected, axis=-1, keepdims=True)
    return np.squeeze(maximum, axis=-1) + np.log(np.exp(selected - maximum).sum(axis=-1))


def truth_margin(
    logits: np.ndarray, true_token_ids: Sequence[int], false_token_ids: Sequence[int]
) -> np.ndarray:
    return class_logit(logits, true_token_ids) - class_logit(logits, false_token_ids)


def expected_margin(raw_truth_margin: np.ndarray, expected_true: np.ndarray) -> np.ndarray:
    signs = np.where(expected_true, 1.0, -1.0)
    return raw_truth_margin * signs


def source_oriented_margin(raw_truth_margin: np.ndarray, source_true: np.ndarray) -> np.ndarray:
    return expected_margin(raw_truth_margin, source_true)


def normalized_restoration(
    patched_source_margin: np.ndarray,
    target_source_margin: np.ndarray,
    source_source_margin: np.ndarray,
    epsilon: float = 1e-08,
) -> np.ndarray:
    denominator = source_source_margin - target_source_margin
    result = np.full_like(denominator, np.nan, dtype=np.float64)
    valid = np.abs(denominator) > epsilon
    result[valid] = (patched_source_margin[valid] - target_source_margin[valid]) / denominator[
        valid
    ]
    return result
