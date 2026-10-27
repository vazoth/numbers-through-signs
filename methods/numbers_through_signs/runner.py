from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from .config import ExperimentConfig, ModelSpec
from .metrics import expected_margin, normalized_restoration, source_oriented_margin, truth_margin
from .model_runtime import ModelRuntime, batched, write_json
from .stimuli import BehaviorTrial, CausalTrial, behavior_trials, causal_trials


def _margins(runtime: ModelRuntime, logits: np.ndarray, tasks: list[str]) -> np.ndarray:
    if len(logits) != len(tasks):
        raise ValueError("Each logit row must have a task")
    result = np.empty(len(tasks), dtype=np.float64)
    task_array = np.asarray(tasks)
    for task in sorted(set(tasks)):
        mask = task_array == task
        verbalizers = runtime.verbalizers[task]
        result[mask] = truth_margin(logits[mask], verbalizers.true, verbalizers.false)
    return result


def _select_trials(
    rows: list[Any],
    *,
    split: str | None = None,
    template: str | None = None,
    source_format: str | None = None,
    target_format: str | None = None,
) -> list[Any]:
    selected = []
    for row in rows:
        if split is not None and row.split != split:
            continue
        if template is not None and row.template_id != template:
            continue
        if source_format is not None and getattr(row, "source_format", None) != source_format:
            continue
        if target_format is not None and getattr(row, "target_format", None) != target_format:
            continue
        selected.append(row)
    return selected


def run_behavior(
    config: ExperimentConfig,
    spec: ModelSpec,
    *,
    output_dir: Path,
    split: str | None = None,
    template: str | None = None,
) -> Path:
    trials: list[BehaviorTrial] = _select_trials(
        list(behavior_trials(config)), split=split, template=template
    )
    if not trials:
        raise ValueError("No behavioral trials matched the requested filters")
    runtime = ModelRuntime(spec, config)
    rows: list[dict[str, Any]] = []
    try:
        carrier = runtime.validate_carrier([trial.prefix for trial in trials])
        if not carrier["is_single_shared_carrier"]:
            raise RuntimeError(f"Carrier is not shared: {carrier}")
        trial_batches = list(batched(trials, spec.microbatch_size))
        for batch in trial_batches:
            logits = runtime.final_logits([trial.prompt for trial in batch])
            raw_margins = _margins(runtime, logits, [trial.task for trial in batch])
            expected = np.array([trial.expected_true for trial in batch], dtype=bool)
            oriented = expected_margin(raw_margins, expected)
            for trial, raw, margin in zip(batch, raw_margins, oriented, strict=True):
                row = trial.to_dict()
                row.update(
                    {
                        "model": spec.short_name,
                        "resolved_revision": runtime.commit_hash,
                        "readout": "full",
                        "raw_truth_margin": float(raw),
                        "expected_margin": float(margin),
                        "correct": bool(margin > 0),
                    }
                )
                rows.append(row)
        model_dir = output_dir / spec.short_name
        model_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_".join((part for part in (split, template) if part)) or "all"
        result_path = model_dir / f"behavior_{'full'}_{suffix}.parquet"
        pd.DataFrame(rows).to_parquet(result_path, index=False)
        manifest = runtime.manifest() | {
            "kind": "behavior",
            "filters": {"split": split, "template": template, "readout": "full"},
            "trial_count": len(rows),
            "carrier_validation": carrier,
            "result_path": str(result_path),
        }
        write_json(model_dir / f"behavior_{'full'}_{suffix}.manifest.json", manifest)
        return result_path
    finally:
        runtime.close()


def _compute_prompt_margins(
    runtime: ModelRuntime, prompts: list[str], tasks: list[str], batch_size: int
) -> dict[str, float]:
    if len(prompts) != len(tasks):
        raise ValueError("Each prompt must have a task")
    prompt_tasks: dict[str, str] = {}
    for prompt, task in zip(prompts, tasks, strict=True):
        previous = prompt_tasks.setdefault(prompt, task)
        if previous != task:
            raise ValueError("The same prompt cannot belong to multiple tasks")
    unique = list(prompt_tasks)
    margins: dict[str, float] = {}
    for batch in batched(unique, batch_size):
        logits = runtime.final_logits(batch)
        values = _margins(runtime, logits, [prompt_tasks[prompt] for prompt in batch])
        margins.update({prompt: float(value) for prompt, value in zip(batch, values, strict=True)})
    return margins


def _cache_prefixes(
    runtime: ModelRuntime, prefixes: list[str], batch_size: int
) -> tuple[list[str], torch.Tensor]:
    unique = list(dict.fromkeys(prefixes))
    chunks = []
    for batch in batched(unique, batch_size):
        chunks.append(runtime.prefix_representations(batch))
    return (unique, torch.cat(chunks, dim=0))


def run_causal(
    config: ExperimentConfig,
    spec: ModelSpec,
    *,
    output_dir: Path,
    split: str | None = None,
    template: str | None = None,
    source_format: str | None = None,
    target_format: str | None = None,
    sites: list[int] | None = None,
) -> tuple[Path, Path]:
    trial_source = causal_trials(config)
    trials: list[CausalTrial] = _select_trials(
        list(trial_source),
        split=split,
        template=template,
        source_format=source_format,
        target_format=target_format,
    )
    if not trials:
        raise ValueError("No causal trials matched the requested filters")
    runtime = ModelRuntime(spec, config)
    model_dir = output_dir / spec.short_name
    model_dir.mkdir(parents=True, exist_ok=True)
    suffix_parts = [part for part in (split, template, source_format, target_format) if part]
    suffix = "_".join(suffix_parts) or "all"
    suffix += "_quantity_trajectory"
    try:
        carrier = runtime.validate_carrier(
            [trial.source_prefix for trial in trials] + [trial.target_prefix for trial in trials]
        )
        if not carrier["is_single_shared_carrier"]:
            raise RuntimeError(f"Carrier is not shared: {carrier}")
        all_prompts = [trial.source_prompt for trial in trials] + [
            trial.target_prompt for trial in trials
        ]
        all_tasks = [trial.task for trial in trials] * 2
        prompt_margins = _compute_prompt_margins(
            runtime, all_prompts, all_tasks, spec.microbatch_size
        )
        baseline_rows: list[dict[str, Any]] = []
        eligible: list[CausalTrial] = []
        for trial in trials:
            source_raw = prompt_margins[trial.source_prompt]
            target_raw = prompt_margins[trial.target_prompt]
            source_expected = source_raw if trial.source_expected_true else -source_raw
            target_expected = target_raw if trial.target_expected_true else -target_raw
            is_eligible = source_expected > 0 and target_expected > 0
            if is_eligible:
                eligible.append(trial)
            baseline_rows.append(
                trial.to_dict()
                | {
                    "model": spec.short_name,
                    "resolved_revision": runtime.commit_hash,
                    "source_raw_truth_margin": source_raw,
                    "target_raw_truth_margin": target_raw,
                    "source_expected_margin": source_expected,
                    "target_expected_margin": target_expected,
                    "eligible": is_eligible,
                }
            )
        baseline_path = model_dir / f"causal_baselines_{suffix}.parquet"
        pd.DataFrame(baseline_rows).to_parquet(baseline_path, index=False)
        if not eligible:
            raise RuntimeError("No eligible source/target pairs survived behavioral gating")
        representation_prefixes = [trial.source_prefix for trial in eligible]
        prefix_order, representations = _cache_prefixes(
            runtime, representation_prefixes, spec.microbatch_size
        )
        prefix_index = {prefix: index for index, prefix in enumerate(prefix_order)}
        selected_sites = sites if sites is not None else list(range(runtime.num_sites))
        if any((site < 0 or site >= runtime.num_sites for site in selected_sites)):
            raise ValueError(f"Requested sites must be inside 0..{runtime.num_sites - 1}")
        result_rows: list[dict[str, Any]] = []
        for site in selected_sites:
            intervention_batches = list(batched(eligible, spec.microbatch_size))
            for batch in intervention_batches:
                source_states = torch.stack(
                    [representations[prefix_index[trial.source_prefix]] for trial in batch], dim=0
                )
                logits = runtime.trajectory_patched_final_logits(
                    [trial.target_prompt for trial in batch],
                    [trial.target_prefix for trial in batch],
                    site,
                    source_states,
                )
                patched_raw = _margins(runtime, logits, [trial.task for trial in batch])
                source_true = np.array([trial.source_expected_true for trial in batch], dtype=bool)
                patched_source = source_oriented_margin(patched_raw, source_true)
                source_source = np.array(
                    [
                        prompt_margins[trial.source_prompt]
                        if trial.source_expected_true
                        else -prompt_margins[trial.source_prompt]
                        for trial in batch
                    ],
                    dtype=np.float64,
                )
                target_source = np.array(
                    [
                        prompt_margins[trial.target_prompt]
                        if trial.source_expected_true
                        else -prompt_margins[trial.target_prompt]
                        for trial in batch
                    ],
                    dtype=np.float64,
                )
                restoration = normalized_restoration(patched_source, target_source, source_source)
                for trial, raw, source_margin, restore in zip(
                    batch, patched_raw, patched_source, restoration, strict=True
                ):
                    result_rows.append(
                        trial.to_dict()
                        | {
                            "model": spec.short_name,
                            "resolved_revision": runtime.commit_hash,
                            "site": site,
                            "intervention": "quantity-trajectory",
                            "relative_depth": site / max(runtime.num_sites - 1, 1),
                            "patched_raw_truth_margin": float(raw),
                            "patched_source_margin": float(source_margin),
                            "source_source_margin": float(
                                prompt_margins[trial.source_prompt]
                                if trial.source_expected_true
                                else -prompt_margins[trial.source_prompt]
                            ),
                            "target_source_margin": float(
                                prompt_margins[trial.target_prompt]
                                * (1.0 if trial.source_expected_true else -1.0)
                            ),
                            "normalized_restoration": float(restore),
                            "iia": bool(source_margin > 0),
                        }
                    )
        result_path = model_dir / f"causal_{suffix}.parquet"
        pd.DataFrame(result_rows).to_parquet(result_path, index=False)
        representation_path = (
            output_dir / "cache" / spec.short_name / f"representations_{suffix}.pt"
        )
        representation_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "prefixes": prefix_order,
                "states": representations,
                "resolved_revision": runtime.commit_hash,
            },
            representation_path,
        )
        manifest = runtime.manifest() | {
            "kind": "causal",
            "filters": {
                "split": split,
                "template": template,
                "source_format": source_format,
                "target_format": target_format,
                "sites": selected_sites,
                "intervention": "quantity-trajectory",
                "control": "counterfactual",
            },
            "requested_trial_count": len(trials),
            "eligible_trial_count": len(eligible),
            "eligible_fraction": len(eligible) / len(trials),
            "result_row_count": len(result_rows),
            "carrier_validation": carrier,
            "result_path": str(result_path),
            "baseline_path": str(baseline_path),
            "representation_cache_path": str(representation_path),
        }
        write_json(model_dir / f"causal_{suffix}.manifest.json", manifest)
        return (baseline_path, result_path)
    finally:
        runtime.close()


def run_representations(
    config: ExperimentConfig, spec: ModelSpec, *, output_dir: Path
) -> tuple[Path, ...]:
    trials = list(behavior_trials(config))
    unique: dict[str, BehaviorTrial] = {}
    for trial in trials:
        unique.setdefault(trial.prefix, trial)
    prefixes = list(unique)
    metadata_rows: list[dict[str, Any]] = []
    for row_index, prefix in enumerate(prefixes):
        trial = unique[prefix]
        metadata_rows.append(
            {
                "row_index": row_index,
                "prefix": prefix,
                "split": trial.split,
                "template_family": "_".join(trial.template_id.split("_")[1:]),
                "format": trial.format,
                "value": trial.value,
                "variant": trial.variant,
            }
        )
    runtime = ModelRuntime(spec, config)
    model_dir = output_dir / spec.short_name
    cache_dir = output_dir / "cache" / spec.short_name
    model_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        carrier = runtime.validate_carrier(prefixes)
        if not carrier["is_single_shared_carrier"]:
            raise RuntimeError(f"Carrier is not shared: {carrier}")
        for position in ("quantity",):
            chunks = []
            for batch in batched(prefixes, spec.microbatch_size):
                chunks.append(runtime.prefix_representations(batch))
            states = torch.cat(chunks, dim=0)
            path = cache_dir / f"representations_all_{position}.pt"
            torch.save(
                {"states": states, "position": position, "resolved_revision": runtime.commit_hash},
                path,
            )
            paths.append(path)
            del states
        metadata_path = model_dir / "representations_metadata.parquet"
        pd.DataFrame(metadata_rows).to_parquet(metadata_path, index=False)
        paths.append(metadata_path)
        manifest = runtime.manifest() | {
            "kind": "representations",
            "positions": ("quantity",),
            "unique_prefix_count": len(prefixes),
            "carrier_validation": carrier,
            "paths": [str(path) for path in paths],
        }
        manifest_path = model_dir / "representations.manifest.json"
        write_json(manifest_path, manifest)
        paths.append(manifest_path)
        return tuple(paths)
    finally:
        runtime.close()
