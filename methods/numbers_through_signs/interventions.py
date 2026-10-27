from __future__ import annotations
import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from .config import ExperimentConfig, ModelSpec
from .metrics import normalized_restoration, truth_margin
from .model_runtime import ModelRuntime, batched, write_json
from .representation_analysis import (
    deterministic_random_bases,
    fit_symbol_cardinal_subspace,
    load_representation_payload,
    projected_replacement,
)
from .equality_stimuli import (
    EqualityCausalTrial,
    InterventionConfig,
    equality_behavior_trials,
    equality_causal_trials,
)
from .stimuli import TEMPLATES, CausalTrial, causal_trials, identity_trials

Trial = CausalTrial | EqualityCausalTrial


def config_with_equality(
    config: ExperimentConfig, settings: InterventionConfig
) -> ExperimentConfig:
    verbalizers = dict(config.class_verbalizers)
    verbalizers["equality"] = dict(settings.equality_verbalizers)
    return replace(config, class_verbalizers=verbalizers)


def ensure_output_path(path: Path, output_root: Path) -> None:
    resolved_path = path.resolve()
    resolved_root = output_root.resolve()
    if resolved_path != resolved_root and resolved_root not in resolved_path.parents:
        raise ValueError(f"Output escapes {resolved_root}: {resolved_path}")


def _prepare_output(path: Path, root: Path, *, overwrite: bool) -> None:
    ensure_output_path(path, root)
    if path.exists() and (not overwrite):
        raise FileExistsError(f"Refusing to overwrite existing measurement output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def _task_margins(runtime: ModelRuntime, logits: np.ndarray, task: str) -> np.ndarray:
    verbalizers = runtime.verbalizers[task]
    return truth_margin(logits, verbalizers.true, verbalizers.false)


def _prompt_margins(
    runtime: ModelRuntime, prompt_tasks: dict[str, str], batch_size: int
) -> dict[str, float]:
    prompts = list(prompt_tasks)
    result: dict[str, float] = {}
    for batch in batched(prompts, batch_size):
        logits = runtime.final_logits(batch)
        tasks = [prompt_tasks[prompt] for prompt in batch]
        raw = np.empty(len(batch), dtype=np.float64)
        task_array = np.asarray(tasks)
        for task in sorted(set(tasks)):
            mask = task_array == task
            raw[mask] = _task_margins(runtime, logits[mask], task)
        result.update({prompt: float(margin) for prompt, margin in zip(batch, raw, strict=True)})
    return result


def _balanced_accuracy(frame: pd.DataFrame) -> float:
    per_class = frame.groupby("expected_true").correct.mean()
    if set(per_class.index) != {False, True}:
        return math.nan
    return float(per_class.mean())


def run_equality_behavior(
    config: ExperimentConfig,
    settings: InterventionConfig,
    spec: ModelSpec,
    *,
    overwrite: bool = False,
) -> Path:
    config = config_with_equality(config, settings)
    model_dir = settings.output_dir / spec.short_name
    output = model_dir / "equality_behavior.parquet"
    manifest_path = model_dir / "equality_behavior.manifest.json"
    _prepare_output(output, settings.output_dir, overwrite=overwrite)
    _prepare_output(manifest_path, settings.output_dir, overwrite=overwrite)
    trials = equality_behavior_trials(config, settings)
    runtime = ModelRuntime(spec, config)
    rows: list[dict[str, Any]] = []
    try:
        verbalizers = runtime.verbalizers["equality"]
        if len(verbalizers.true) != 1 or len(verbalizers.false) != 1:
            raise RuntimeError("Equality yes/no verbalizers must each be one token")
        for batch in batched(trials, spec.microbatch_size):
            logits = runtime.final_logits([trial.prompt for trial in batch])
            raw = _task_margins(runtime, logits, "equality")
            for trial, margin in zip(batch, raw, strict=True):
                expected_margin = margin if trial.expected_true else -margin
                rows.append(
                    trial.to_dict()
                    | {
                        "model": spec.short_name,
                        "resolved_revision": runtime.commit_hash,
                        "backend": runtime.placement,
                        "raw_truth_margin": float(margin),
                        "expected_margin": float(expected_margin),
                        "correct": bool(expected_margin > 0),
                    }
                )
        pd.DataFrame(rows).to_parquet(output, index=False)
        write_json(
            manifest_path,
            runtime.manifest()
            | {"kind": "equality-behavior", "trial_count": len(rows), "result_path": str(output)},
        )
        return output
    finally:
        runtime.close()


def _trials_for_operation(
    config: ExperimentConfig,
    settings: InterventionConfig,
    *,
    task: str,
    template_id: str,
    control: str,
    directions: Sequence[tuple[str, str]] | None = None,
) -> list[Trial]:
    if task == "equality":
        trials: list[Trial] = equality_causal_trials(
            config, settings, template_id=template_id, control=control
        )
    elif task in ("comparison", "parity"):
        source = causal_trials(config) if control == "counterfactual" else identity_trials(config)
        trials = [
            trial for trial in source if trial.task == task and trial.template_id == template_id
        ]
    else:
        raise ValueError("task must be 'comparison', 'parity', or 'equality'")
    if directions is not None:
        allowed = set(directions)
        trials = [
            trial for trial in trials if (trial.source_format, trial.target_format) in allowed
        ]
    return trials


def _require_template_admission(
    settings: InterventionConfig, spec: ModelSpec, *, task: str, template_id: str
) -> None:
    if task == "comparison" and template_id == "cmp_eval_number":
        return
    if task == "equality":
        admission_path = settings.output_dir / spec.short_name / "equality_admission.parquet"
    elif task in ("comparison", "parity"):
        admission_path = settings.output_dir / spec.short_name / f"{task}_admission.parquet"
    else:
        raise ValueError("task must be comparison, parity, or equality")
    if not admission_path.exists():
        raise FileNotFoundError(f"{task} admission must be frozen before intervention")
    admission = pd.read_parquet(admission_path)
    admitted = bool(admission.loc[admission.template_id.eq(template_id), "admitted"].any())
    if not admitted:
        raise RuntimeError(f"{task} template {template_id} is not admitted")


def screen_equality_templates(
    config: ExperimentConfig,
    settings: InterventionConfig,
    spec: ModelSpec,
    *,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    config = config_with_equality(config, settings)
    model_dir = settings.output_dir / spec.short_name
    behavior_path = model_dir / "equality_behavior.parquet"
    if not behavior_path.exists():
        raise FileNotFoundError("Run equality behavior before admission screening")
    summary_path = model_dir / "equality_admission.parquet"
    baseline_path = model_dir / "equality_screen_baselines.parquet"
    manifest_path = model_dir / "equality_admission.json"
    for path in (summary_path, baseline_path, manifest_path):
        _prepare_output(path, settings.output_dir, overwrite=overwrite)
    behavior = pd.read_parquet(behavior_path)
    runtime = ModelRuntime(spec, config)
    baseline_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    gate = settings.equality_gate
    try:
        for template in settings.equality_templates:
            template_behavior = behavior[behavior.template_id.eq(template.id)]
            format_metrics: dict[str, dict[str, float]] = {}
            for format_name in ("digit", "word", "run", "fixed"):
                subset = template_behavior[template_behavior.format.eq(format_name)]
                class_accuracy = subset.groupby("expected_true").correct.mean()
                format_metrics[format_name] = {
                    "balanced_accuracy": _balanced_accuracy(subset),
                    "false_accuracy": float(class_accuracy.get(False, math.nan)),
                    "true_accuracy": float(class_accuracy.get(True, math.nan)),
                }
            trials = _trials_for_operation(
                config,
                settings,
                task="equality",
                template_id=template.id,
                control="counterfactual",
                directions=settings.primary_directions,
            )
            prompt_tasks = {
                prompt: "equality"
                for trial in trials
                for prompt in (trial.source_prompt, trial.target_prompt)
            }
            margins = _prompt_margins(runtime, prompt_tasks, spec.microbatch_size)
            eligible_counts: dict[tuple[str, str], int] = {
                direction: 0 for direction in settings.primary_directions
            }
            requested_counts: dict[tuple[str, str], int] = {
                direction: 0 for direction in settings.primary_directions
            }
            for trial in trials:
                source_expected = margins[trial.source_prompt] * (
                    1.0 if trial.source_expected_true else -1.0
                )
                target_expected = margins[trial.target_prompt] * (
                    1.0 if trial.target_expected_true else -1.0
                )
                eligible = source_expected > 0 and target_expected > 0
                direction = (trial.source_format, trial.target_format)
                requested_counts[direction] += 1
                eligible_counts[direction] += int(eligible)
                baseline_rows.append(
                    trial.to_dict()
                    | {
                        "model": spec.short_name,
                        "source_raw_truth_margin": margins[trial.source_prompt],
                        "target_raw_truth_margin": margins[trial.target_prompt],
                        "source_expected_margin": source_expected,
                        "target_expected_margin": target_expected,
                        "eligible": eligible,
                    }
                )
            min_eligible = min(eligible_counts.values())
            digit_ok = format_metrics["digit"]["balanced_accuracy"] >= float(
                gate["digit_min_balanced_accuracy"]
            )
            word_ok = format_metrics["word"]["balanced_accuracy"] >= float(
                gate["word_min_balanced_accuracy"]
            )
            run_ok = format_metrics["run"]["balanced_accuracy"] >= float(
                gate["run_min_balanced_accuracy"]
            ) and min(
                format_metrics["run"]["false_accuracy"], format_metrics["run"]["true_accuracy"]
            ) > float(gate["each_class_min_accuracy_exclusive"])
            eligible_ok = min_eligible >= int(gate["principal_direction_min_eligible"])
            admitted = digit_ok and word_ok and run_ok and eligible_ok
            summary_rows.append(
                {
                    "model": spec.short_name,
                    "template_id": template.id,
                    "admitted": admitted,
                    "digit_balanced_accuracy": format_metrics["digit"]["balanced_accuracy"],
                    "word_balanced_accuracy": format_metrics["word"]["balanced_accuracy"],
                    "run_balanced_accuracy": format_metrics["run"]["balanced_accuracy"],
                    "run_false_accuracy": format_metrics["run"]["false_accuracy"],
                    "run_true_accuracy": format_metrics["run"]["true_accuracy"],
                    "minimum_principal_direction_eligible": min_eligible,
                    "principal_direction_eligible": json.dumps(
                        {
                            f"{source}->{target}": eligible_counts[source, target]
                            for source, target in settings.primary_directions
                        },
                        sort_keys=True,
                    ),
                    "principal_direction_requested": json.dumps(
                        {
                            f"{source}->{target}": requested_counts[source, target]
                            for source, target in settings.primary_directions
                        },
                        sort_keys=True,
                    ),
                }
            )
        pd.DataFrame(baseline_rows).to_parquet(baseline_path, index=False)
        summary = pd.DataFrame(summary_rows)
        summary.to_parquet(summary_path, index=False)
        write_json(
            manifest_path,
            runtime.manifest()
            | {
                "kind": "equality-admission",
                "gate": gate,
                "admitted_templates": summary.loc[summary.admitted, "template_id"].tolist(),
                "screening_only_no_interventions": True,
                "summary_path": str(summary_path),
                "baseline_path": str(baseline_path),
            },
        )
        return (summary_path, baseline_path)
    finally:
        runtime.close()


def screen_base_templates(
    config: ExperimentConfig,
    settings: InterventionConfig,
    spec: ModelSpec,
    *,
    task: str,
    behavior_path: Path,
    overwrite: bool = False,
) -> tuple[Path, Path]:
    """Freeze comparison/parity admission using behavior only."""
    if task not in ("comparison", "parity"):
        raise ValueError("Base-template screening supports comparison or parity")
    model_dir = settings.output_dir / spec.short_name
    summary_path = model_dir / f"{task}_admission.parquet"
    baseline_path = model_dir / f"{task}_screen_baselines.parquet"
    manifest_path = model_dir / f"{task}_admission.json"
    for path in (summary_path, baseline_path, manifest_path):
        _prepare_output(path, settings.output_dir, overwrite=overwrite)
    behavior = pd.read_parquet(behavior_path)
    behavior = behavior[behavior.task.eq(task)].copy()
    if behavior.empty:
        raise ValueError(f"No {task} behavior rows in {behavior_path}")
    prompt_margin = behavior.groupby("prompt", sort=False).raw_truth_margin.first().to_dict()
    gate = settings.prompt_admission_gate
    baseline_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    templates = [
        template for template in TEMPLATES if template.split == "eval" and template.task == task
    ]
    for template in templates:
        template_behavior = behavior[behavior.template_id.eq(template.id)]
        format_metrics: dict[str, dict[str, float]] = {}
        for format_name in config.formats:
            subset = template_behavior[template_behavior.format.eq(format_name)]
            class_accuracy = subset.groupby("expected_true").correct.mean()
            format_metrics[format_name] = {
                "balanced_accuracy": _balanced_accuracy(subset),
                "false_accuracy": float(class_accuracy.get(False, math.nan)),
                "true_accuracy": float(class_accuracy.get(True, math.nan)),
            }
        trials = _trials_for_operation(
            config,
            settings,
            task=task,
            template_id=template.id,
            control="counterfactual",
            directions=settings.primary_directions,
        )
        eligible_counts = {direction: 0 for direction in settings.primary_directions}
        requested_counts = {direction: 0 for direction in settings.primary_directions}
        for trial in trials:
            try:
                source_raw = float(prompt_margin[trial.source_prompt])
                target_raw = float(prompt_margin[trial.target_prompt])
            except KeyError as error:
                raise KeyError(
                    f"Behavioral screen lacks a causal endpoint prompt for {template.id}"
                ) from error
            source_expected = source_raw * (1.0 if trial.source_expected_true else -1.0)
            target_expected = target_raw * (1.0 if trial.target_expected_true else -1.0)
            eligible = source_expected > 0 and target_expected > 0
            direction = (trial.source_format, trial.target_format)
            requested_counts[direction] += 1
            eligible_counts[direction] += int(eligible)
            baseline_rows.append(
                trial.to_dict()
                | {
                    "model": spec.short_name,
                    "source_raw_truth_margin": source_raw,
                    "target_raw_truth_margin": target_raw,
                    "source_expected_margin": source_expected,
                    "target_expected_margin": target_expected,
                    "eligible": eligible,
                }
            )
        minimum_eligible = min(eligible_counts.values())
        digit_ok = format_metrics["digit"]["balanced_accuracy"] >= float(
            gate["digit_min_balanced_accuracy"]
        )
        word_ok = format_metrics["word"]["balanced_accuracy"] >= float(
            gate["word_min_balanced_accuracy"]
        )
        run_ok = format_metrics["run"]["balanced_accuracy"] >= float(
            gate["run_min_balanced_accuracy"]
        ) and min(
            format_metrics["run"]["false_accuracy"], format_metrics["run"]["true_accuracy"]
        ) > float(gate["each_class_min_accuracy_exclusive"])
        eligible_ok = minimum_eligible >= int(gate["principal_direction_min_eligible"])
        admitted = digit_ok and word_ok and run_ok and eligible_ok
        row: dict[str, Any] = {
            "model": spec.short_name,
            "task": task,
            "template_id": template.id,
            "admitted": admitted,
            "minimum_principal_direction_eligible": minimum_eligible,
            "principal_direction_eligible": json.dumps(
                {
                    f"{source}->{target}": eligible_counts[source, target]
                    for source, target in settings.primary_directions
                },
                sort_keys=True,
            ),
            "principal_direction_requested": json.dumps(
                {
                    f"{source}->{target}": requested_counts[source, target]
                    for source, target in settings.primary_directions
                },
                sort_keys=True,
            ),
        }
        for format_name, metrics in format_metrics.items():
            for metric, value in metrics.items():
                row[f"{format_name}_{metric}"] = value
        summary_rows.append(row)
    pd.DataFrame(baseline_rows).to_parquet(baseline_path, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_parquet(summary_path, index=False)
    write_json(
        manifest_path,
        {
            "kind": f"{task}-admission",
            "model_id": spec.id,
            "model_short_name": spec.short_name,
            "requested_revision": spec.revision,
            "behavior_path": str(behavior_path),
            "gate": gate,
            "admitted_templates": summary.loc[summary.admitted, "template_id"].tolist(),
            "screening_only_no_interventions": True,
            "summary_path": str(summary_path),
            "baseline_path": str(baseline_path),
        },
    )
    return (summary_path, baseline_path)


def _wrong_donor_map(trials: Sequence[Trial]) -> dict[str, Trial]:
    ordered = sorted(trials, key=lambda trial: trial.trial_id)
    result: dict[str, Trial] = {}
    for trial in trials:
        candidates = [
            candidate
            for candidate in ordered
            if candidate.source_format == trial.source_format
            and candidate.source_value not in (trial.source_value, trial.target_value)
            and (candidate.source_expected_true == trial.target_expected_true)
        ]
        if not candidates:
            candidates = [
                candidate
                for candidate in ordered
                if candidate.source_value not in (trial.source_value, trial.target_value)
            ]
        if not candidates:
            raise RuntimeError(f"No wrong-value donor for {trial.trial_id}")
        result[trial.trial_id] = candidates[0]
    return result


def _cache_prefix_states(
    runtime: ModelRuntime, prefixes: Sequence[str], batch_size: int
) -> tuple[list[str], torch.Tensor]:
    unique = list(dict.fromkeys(prefixes))
    chunks = [runtime.prefix_representations(batch) for batch in batched(unique, batch_size)]
    return (unique, torch.cat(chunks, dim=0))


def _resolve_start_site(
    settings: InterventionConfig, spec: ModelSpec, *, start_site: int | None, all_sites: bool
) -> int:
    if start_site is not None:
        return start_site
    if all_sites:
        return 1
    return settings.frozen_sites[spec.short_name]


def run_quantity_windows(
    config: ExperimentConfig,
    settings: InterventionConfig,
    spec: ModelSpec,
    *,
    task: str,
    template_id: str,
    control: str = "counterfactual",
    donor: str = "source",
    windows: Sequence[int | str] | None = None,
    start_site: int | None = None,
    directions: Sequence[tuple[str, str]] | None = None,
    overwrite: bool = False,
    all_sites: bool = False,
) -> Path:
    if donor not in ("source", "wrong", "exact-target"):
        raise ValueError("donor must be source, wrong, or exact-target")
    config = config_with_equality(config, settings)
    _require_template_admission(settings, spec, task=task, template_id=template_id)
    selected_directions = tuple(directions or settings.primary_directions)
    trials = _trials_for_operation(
        config,
        settings,
        task=task,
        template_id=template_id,
        control=control,
        directions=selected_directions,
    )
    if not trials:
        raise ValueError("No trials selected")
    selected_windows = tuple(windows or settings.quantity_windows)
    site = _resolve_start_site(settings, spec, start_site=start_site, all_sites=all_sites)
    model_dir = settings.output_dir / spec.short_name
    all_format_directions = set(itertools.product(config.formats, repeat=2))
    direction_suffix = "_all-formats" if set(selected_directions) == all_format_directions else ""
    suffix = (
        f"{task}_{template_id}_{control}_{donor}{direction_suffix}_all-sites"
        if all_sites
        else f"{task}_{template_id}_{control}_{donor}{direction_suffix}_site{site}"
    )
    output = model_dir / f"quantity_windows_{suffix}.parquet"
    manifest_path = model_dir / f"quantity_windows_{suffix}.manifest.json"
    _prepare_output(output, settings.output_dir, overwrite=overwrite)
    _prepare_output(manifest_path, settings.output_dir, overwrite=overwrite)
    runtime = ModelRuntime(spec, config)
    try:
        prompt_tasks = {
            prompt: task
            for trial in trials
            for prompt in (trial.source_prompt, trial.target_prompt)
        }
        margins = _prompt_margins(runtime, prompt_tasks, spec.microbatch_size)
        wrong_map = _wrong_donor_map(trials) if donor == "wrong" else {}
        exact_states: dict[str, torch.Tensor] = {}
        exact_raw: dict[str, float] = {}
        if donor == "exact-target":
            for batch in batched(trials, spec.microbatch_size):
                states, logits = runtime.prompt_representations(
                    [trial.target_prompt for trial in batch],
                    [trial.target_prefix for trial in batch],
                )
                raw = _task_margins(runtime, logits, task)
                for trial, state, margin in zip(batch, states, raw, strict=True):
                    exact_states[trial.trial_id] = state
                    exact_raw[trial.trial_id] = float(margin)
        else:
            donor_prefixes = [
                wrong_map[trial.trial_id].source_prefix if donor == "wrong" else trial.source_prefix
                for trial in trials
            ]
            prefix_order, cached = _cache_prefix_states(
                runtime, donor_prefixes, spec.microbatch_size
            )
            prefix_index = {prefix: index for index, prefix in enumerate(prefix_order)}
        rows: list[dict[str, Any]] = []
        evaluated_sites = tuple(range(1, runtime.num_sites)) if all_sites else (site,)
        for evaluated_site in evaluated_sites:
            for window in selected_windows:
                for batch in batched(trials, spec.microbatch_size):
                    if donor == "exact-target":
                        source_states = torch.stack(
                            [exact_states[trial.trial_id] for trial in batch], dim=0
                        )
                    else:
                        source_states = torch.stack(
                            [
                                cached[
                                    prefix_index[
                                        wrong_map[trial.trial_id].source_prefix
                                        if donor == "wrong"
                                        else trial.source_prefix
                                    ]
                                ]
                                for trial in batch
                            ],
                            dim=0,
                        )
                    logits = runtime.window_patched_final_logits(
                        [trial.target_prompt for trial in batch],
                        [trial.target_prefix for trial in batch],
                        evaluated_site,
                        window,
                        source_states,
                    )
                    patched_raw = _task_margins(runtime, logits, task)
                    for trial, raw in zip(batch, patched_raw, strict=True):
                        source_expected = margins[trial.source_prompt] * (
                            1.0 if trial.source_expected_true else -1.0
                        )
                        target_expected = margins[trial.target_prompt] * (
                            1.0 if trial.target_expected_true else -1.0
                        )
                        eligible = source_expected > 0 and target_expected > 0
                        source_sign = 1.0 if trial.source_expected_true else -1.0
                        target_sign = 1.0 if trial.target_expected_true else -1.0
                        patched_source = float(raw * source_sign)
                        baseline_target_raw = (
                            exact_raw[trial.trial_id]
                            if donor == "exact-target"
                            else margins[trial.target_prompt]
                        )
                        target_source = float(baseline_target_raw * source_sign)
                        restoration = normalized_restoration(
                            np.array([patched_source]),
                            np.array([target_source]),
                            np.array([source_expected]),
                        )[0]
                        wrong_trial = wrong_map.get(trial.trial_id)
                        rows.append(
                            trial.to_dict()
                            | {
                                "model": spec.short_name,
                                "resolved_revision": runtime.commit_hash,
                                "backend": runtime.placement,
                                "site": evaluated_site,
                                "relative_depth": evaluated_site / max(runtime.num_sites - 1, 1),
                                "position": "quantity",
                                "window": str(window),
                                "window_site_count": runtime.num_sites - evaluated_site
                                if window == "to-end"
                                else min(int(window), runtime.num_sites - evaluated_site),
                                "donor": donor,
                                "donor_value": wrong_trial.source_value
                                if wrong_trial is not None
                                else trial.target_value
                                if donor == "exact-target"
                                else trial.source_value,
                                "donor_format": wrong_trial.source_format
                                if wrong_trial is not None
                                else trial.target_format
                                if donor == "exact-target"
                                else trial.source_format,
                                "source_raw_truth_margin": margins[trial.source_prompt],
                                "target_raw_truth_margin": margins[trial.target_prompt],
                                "source_expected_margin": source_expected,
                                "target_expected_margin": target_expected,
                                "eligible": eligible,
                                "patched_raw_truth_margin": float(raw),
                                "patched_source_margin": patched_source,
                                "patched_target_margin": float(raw * target_sign),
                                "source_directed_gain": patched_source - target_source,
                                "normalized_restoration": float(restoration),
                                "iia": bool(patched_source > 0)
                                if eligible and donor != "exact-target"
                                else None,
                                "target_class_preserved": bool(raw * target_sign > 0),
                                "exact_noop_margin_delta": float(raw - baseline_target_raw)
                                if donor == "exact-target"
                                else None,
                            }
                        )
        pd.DataFrame(rows).to_parquet(output, index=False)
        write_json(
            manifest_path,
            runtime.manifest()
            | {
                "kind": "quantity-windows",
                "task": task,
                "template_id": template_id,
                "control": control,
                "donor": donor,
                "directions": selected_directions,
                "windows": selected_windows,
                "site": None if all_sites else site,
                "sites": evaluated_sites,
                "all_sites": all_sites,
                "requested_trial_count": len(trials),
                "row_count": len(rows),
                "result_path": str(output),
            },
        )
        return output
    finally:
        runtime.close()


def run_subspace_interventions(
    config: ExperimentConfig,
    settings: InterventionConfig,
    spec: ModelSpec,
    *,
    task: str,
    template_id: str,
    control: str,
    metadata_path: Path,
    state_path: Path,
    directions: Sequence[tuple[str, str]] | None = None,
    start_site: int | None = None,
    overwrite: bool = False,
) -> Path:
    config = config_with_equality(config, settings)
    _require_template_admission(settings, spec, task=task, template_id=template_id)
    selected_directions = tuple(directions or settings.primary_directions)
    trials = _trials_for_operation(
        config,
        settings,
        task=task,
        template_id=template_id,
        control=control,
        directions=selected_directions,
    )
    if not trials:
        raise ValueError("No subspace trials selected")
    site = start_site if start_site is not None else settings.frozen_sites[spec.short_name]
    metadata, states = load_representation_payload(metadata_path, state_path)
    fitted = fit_symbol_cardinal_subspace(
        states, metadata, site=site, max_rank=int(settings.subspace["max_rank"])
    )
    fitting = metadata.split.eq("dev") & metadata.format.isin(["digit", "word"])
    fit_labels = metadata.loc[fitting, "value"].to_numpy(dtype=np.int64)
    permuted_fit = fit_symbol_cardinal_subspace(
        states,
        metadata,
        site=site,
        max_rank=int(settings.subspace["max_rank"]),
        permuted_fit_labels=np.random.default_rng(settings.seed + site + 10000).permutation(
            fit_labels
        ),
    )
    random_bases = deterministic_random_bases(
        hidden_size=fitted.basis.shape[0],
        rank=fitted.rank,
        count=int(settings.subspace["random_control_count"]),
        seed=settings.seed + site,
    )
    bases: dict[str, np.ndarray | None] = {
        "full-native": None,
        "symbol-cardinal": fitted.basis,
        "top-variance-pca": fitted.pca_basis,
        "permuted-label": permuted_fit.basis,
        **{f"random-{index:02d}": basis for index, basis in enumerate(random_bases)},
    }
    model_dir = settings.output_dir / spec.short_name
    all_format_directions = set(itertools.product(config.formats, repeat=2))
    direction_suffix = "_all-formats" if set(selected_directions) == all_format_directions else ""
    suffix = f"{task}_{template_id}_{control}{direction_suffix}_site{site}"
    output = model_dir / f"subspace_{suffix}.parquet"
    manifest_path = model_dir / f"subspace_{suffix}.manifest.json"
    basis_path = model_dir / f"subspace_{suffix}.npz"
    for path in (output, manifest_path, basis_path):
        _prepare_output(path, settings.output_dir, overwrite=overwrite)
    runtime = ModelRuntime(spec, config)
    try:
        prompt_tasks = {
            prompt: task
            for trial in trials
            for prompt in (trial.source_prompt, trial.target_prompt)
        }
        margins = _prompt_margins(runtime, prompt_tasks, spec.microbatch_size)
        prefixes = [
            prefix for trial in trials for prefix in (trial.source_prefix, trial.target_prefix)
        ]
        prefix_order, prefix_states = _cache_prefix_states(runtime, prefixes, spec.microbatch_size)
        prefix_index = {prefix: index for index, prefix in enumerate(prefix_order)}
        rows: list[dict[str, Any]] = []
        for basis_name, basis in bases.items():
            for batch in batched(trials, spec.microbatch_size):
                source = np.stack(
                    [
                        prefix_states[prefix_index[trial.source_prefix], site].float().numpy()
                        for trial in batch
                    ],
                    axis=0,
                )
                target = np.stack(
                    [
                        prefix_states[prefix_index[trial.target_prefix], site].float().numpy()
                        for trial in batch
                    ],
                    axis=0,
                )
                if basis is None:
                    replacements = source
                else:
                    replacements = np.stack(
                        [
                            projected_replacement(
                                source[index : index + 1],
                                target[index : index + 1],
                                basis,
                                fitted.format_means[trial.source_format],
                                fitted.format_means[trial.target_format],
                            )[0]
                            for index, trial in enumerate(batch)
                        ],
                        axis=0,
                    )
                logits = runtime.patched_final_logits(
                    [trial.target_prompt for trial in batch],
                    [trial.target_prefix for trial in batch],
                    site,
                    torch.from_numpy(replacements),
                )
                raw_margins = _task_margins(runtime, logits, task)
                for trial, raw in zip(batch, raw_margins, strict=True):
                    source_sign = 1.0 if trial.source_expected_true else -1.0
                    target_sign = 1.0 if trial.target_expected_true else -1.0
                    source_expected = margins[trial.source_prompt] * source_sign
                    target_expected = margins[trial.target_prompt] * target_sign
                    eligible = source_expected > 0 and target_expected > 0
                    patched_source = float(raw * source_sign)
                    target_source = float(margins[trial.target_prompt] * source_sign)
                    restoration = normalized_restoration(
                        np.array([patched_source]),
                        np.array([target_source]),
                        np.array([source_expected]),
                    )[0]
                    rows.append(
                        trial.to_dict()
                        | {
                            "model": spec.short_name,
                            "resolved_revision": runtime.commit_hash,
                            "backend": runtime.placement,
                            "site": site,
                            "relative_depth": site / max(runtime.num_sites - 1, 1),
                            "position": "quantity",
                            "window": "1",
                            "window_site_count": 1,
                            "donor": basis_name,
                            "donor_value": trial.source_value,
                            "donor_format": trial.source_format,
                            "subspace_rank": fitted.rank,
                            "source_raw_truth_margin": margins[trial.source_prompt],
                            "target_raw_truth_margin": margins[trial.target_prompt],
                            "source_expected_margin": source_expected,
                            "target_expected_margin": target_expected,
                            "eligible": eligible,
                            "patched_raw_truth_margin": float(raw),
                            "patched_source_margin": patched_source,
                            "patched_target_margin": float(raw * target_sign),
                            "source_directed_gain": patched_source - target_source,
                            "normalized_restoration": float(restoration),
                            "iia": bool(patched_source > 0) if eligible else None,
                            "target_class_preserved": bool(raw * target_sign > 0),
                        }
                    )
        pd.DataFrame(rows).to_parquet(output, index=False)
        np.savez_compressed(
            basis_path,
            symbol_cardinal=fitted.basis,
            top_variance_pca=fitted.pca_basis,
            permuted_label=permuted_fit.basis,
            **{
                name.replace("-", "_"): basis
                for name, basis in bases.items()
                if name.startswith("random-")
            },
        )
        write_json(
            manifest_path,
            runtime.manifest()
            | {
                "kind": "subspace-intervention",
                "task": task,
                "template_id": template_id,
                "control": control,
                "directions": selected_directions,
                "site": site,
                "rank": fitted.rank,
                "fit_row_indices": fitted.fit_row_indices,
                "fit_values": fitted.fit_values,
                "basis_controls": list(bases),
                "requested_trial_count": len(trials),
                "row_count": len(rows),
                "result_path": str(output),
                "basis_path": str(basis_path),
            },
        )
        return output
    finally:
        runtime.close()
