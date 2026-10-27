from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from .probe_analysis import fit_dual_ridge, select_alpha


@dataclass(frozen=True)
class CardinalSubspace:
    basis: np.ndarray
    pca_basis: np.ndarray
    format_means: dict[str, np.ndarray]
    rank: int
    fit_row_indices: tuple[int, ...]
    fit_values: tuple[int, ...]


def leave_one_value_out_predictions(
    states: np.ndarray, metadata: pd.DataFrame, *, site: int, alpha_candidates: tuple[float, ...]
) -> pd.DataFrame:
    if states.ndim != 3:
        raise ValueError("states must have shape [row, site, hidden]")
    if len(metadata) != states.shape[0]:
        raise ValueError("metadata and state rows differ")
    if not 0 <= site < states.shape[1]:
        raise IndexError(site)
    layer = states[:, site].astype(np.float64)
    formats = sorted(metadata.format.unique())
    values = sorted((int(value) for value in metadata.value.unique()))
    dev = metadata.split.eq("dev").to_numpy()
    evaluation = metadata.split.eq("eval").to_numpy()
    rows: list[dict[str, Any]] = []
    for source_format in formats:
        for held_out in values:
            training = (
                dev
                & metadata.format.eq(source_format).to_numpy()
                & metadata.value.ne(held_out).to_numpy()
            )
            train_metadata = metadata.loc[training]
            if held_out in set(train_metadata.value):
                raise AssertionError("Held-out value leaked into probe training")
            alpha = select_alpha(layer[training], train_metadata, alpha_candidates)
            target = train_metadata.value.to_numpy(dtype=np.float64)
            observed = fit_dual_ridge(layer[training], target, alpha)
            for target_format in formats:
                testing = (
                    evaluation
                    & metadata.format.eq(target_format).to_numpy()
                    & metadata.value.eq(held_out).to_numpy()
                )
                if not testing.any():
                    continue
                test_rows = metadata.loc[testing]
                observed_predictions = observed.predict(layer[testing])
                for metadata_row, prediction in zip(
                    test_rows.itertuples(), observed_predictions, strict=True
                ):
                    rows.append(
                        {
                            "row_index": int(metadata_row.row_index),
                            "site": site,
                            "source_format": source_format,
                            "target_format": target_format,
                            "held_out_value": held_out,
                            "template_family": metadata_row.template_family,
                            "variant": int(metadata_row.variant),
                            "alpha": alpha,
                            "train_count": int(training.sum()),
                            "prediction": float(prediction),
                        }
                    )
    return pd.DataFrame(rows)


def _orthonormal_basis(matrix: np.ndarray, rank: int) -> np.ndarray:
    _, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    usable = int(np.sum(singular > np.finfo(np.float64).eps * singular[0]))
    selected_rank = min(rank, usable)
    if selected_rank < 1:
        raise ValueError("No non-zero directions available")
    return vh[:selected_rank].T


def fit_symbol_cardinal_subspace(
    states: np.ndarray,
    metadata: pd.DataFrame,
    *,
    site: int,
    max_rank: int,
    permuted_fit_labels: np.ndarray | None = None,
) -> CardinalSubspace:
    if len(metadata) != states.shape[0]:
        raise ValueError("metadata and state rows differ")
    dev = metadata.split.eq("dev")
    symbols = metadata.format.isin(["digit", "word"])
    fitting = dev & symbols
    fit_metadata = metadata.loc[fitting].copy()
    fit_states = states[fitting.to_numpy(), site].astype(np.float64)
    if set(fit_metadata.format.unique()) != {"digit", "word"}:
        raise ValueError("Both digit and word development states are required")
    all_dev = metadata.loc[dev]
    all_dev_states = states[dev.to_numpy(), site].astype(np.float64)
    format_means = {
        str(format_name): all_dev_states[all_dev.format.eq(format_name).to_numpy()].mean(axis=0)
        for format_name in sorted(all_dev.format.unique())
    }
    centered = fit_states.copy()
    for format_name in ("digit", "word"):
        mask = fit_metadata.format.eq(format_name).to_numpy()
        centered[mask] -= centered[mask].mean(axis=0)
    for template_family in sorted(fit_metadata.template_family.unique()):
        mask = fit_metadata.template_family.eq(template_family).to_numpy()
        centered[mask] -= centered[mask].mean(axis=0)
    if permuted_fit_labels is None:
        fit_labels = fit_metadata.value.to_numpy(dtype=np.int64)
    else:
        fit_labels = np.asarray(permuted_fit_labels, dtype=np.int64)
        if fit_labels.shape != (len(fit_metadata),):
            raise ValueError(
                "permuted_fit_labels must have one label per development digit/word fitting row"
            )
        if sorted(fit_labels.tolist()) != sorted(
            fit_metadata.value.to_numpy(dtype=np.int64).tolist()
        ):
            raise ValueError("Permuted labels must preserve the fitting-label multiset")
    fit_metadata["_fit_label"] = fit_labels
    value_centroids = np.stack(
        [
            centered[fit_metadata["_fit_label"].eq(value).to_numpy()].mean(axis=0)
            for value in sorted(fit_metadata["_fit_label"].unique())
        ],
        axis=0,
    )
    value_centroids -= value_centroids.mean(axis=0)
    rank = min(max_rank, len(fit_metadata["_fit_label"].unique()) - 1, value_centroids.shape[1])
    basis = _orthonormal_basis(value_centroids, rank)
    pca_centered = fit_states - fit_states.mean(axis=0)
    pca_basis = _orthonormal_basis(pca_centered, basis.shape[1])
    return CardinalSubspace(
        basis=basis,
        pca_basis=pca_basis,
        format_means=format_means,
        rank=basis.shape[1],
        fit_row_indices=tuple((int(value) for value in fit_metadata.row_index)),
        fit_values=tuple(sorted((int(value) for value in fit_metadata.value.unique()))),
    )


def deterministic_random_bases(
    *, hidden_size: int, rank: int, count: int, seed: int
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    bases = []
    for _ in range(count):
        matrix = rng.normal(size=(hidden_size, rank))
        q, _ = np.linalg.qr(matrix)
        bases.append(q[:, :rank])
    return tuple(bases)


def projected_replacement(
    source: np.ndarray,
    target: np.ndarray,
    basis: np.ndarray,
    source_mean: np.ndarray,
    target_mean: np.ndarray,
) -> np.ndarray:
    if source.shape != target.shape:
        raise ValueError("source and target state shapes differ")
    difference = source - source_mean - (target - target_mean)
    projected = difference @ basis @ basis.T
    return target + projected


def load_representation_payload(
    metadata_path: Path, state_path: Path
) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = pd.read_parquet(metadata_path).sort_values("row_index")
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    states = payload["states"].float().numpy()
    if len(metadata) != states.shape[0]:
        raise ValueError("Representation payload row mismatch")
    return (metadata, states)
