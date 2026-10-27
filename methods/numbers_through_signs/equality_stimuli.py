from __future__ import annotations
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal
from .config import ExperimentConfig
from .stimuli import CARRIER, Format, stable_id, surface, variants_for


@dataclass(frozen=True)
class EqualityTemplate:
    id: str
    prefix: str
    suffix: str

    def render(self, expression: str, threshold: int) -> tuple[str, str]:
        prefix = self.prefix.format(expression=expression)
        if not prefix.endswith(CARRIER):
            raise ValueError(f"Equality template {self.id} lacks the shared carrier")
        return (prefix, prefix + self.suffix.format(threshold=threshold))


@dataclass(frozen=True)
class InterventionConfig:
    output_dir: Path
    seed: int
    frozen_sites: dict[str, int]
    equality_templates: tuple[EqualityTemplate, ...]
    equality_verbalizers: dict[str, tuple[str, ...]]
    equality_gate: dict[str, float | int]
    quantity_windows: tuple[int | str, ...]
    prompt_admission_gate: dict[str, float | int]
    primary_directions: tuple[tuple[str, str], ...]
    subspace: dict[str, Any]


@dataclass(frozen=True)
class EqualityBehaviorTrial:
    trial_id: str
    split: Literal["eval"]
    task: Literal["equality"]
    template_id: str
    format: Format
    value: int
    variant: int
    threshold: int
    expected_true: bool
    prefix: str
    prompt: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EqualityCausalTrial:
    trial_id: str
    split: Literal["eval"]
    task: Literal["equality"]
    template_id: str
    source_format: Format
    target_format: Format
    source_value: int
    target_value: int
    source_variant: int
    target_variant: int
    threshold: int
    source_expected_true: bool
    target_expected_true: bool
    source_prefix: str
    source_prompt: str
    target_prefix: str
    target_prompt: str
    control: str = "counterfactual"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_intervention_config(path: str | Path) -> InterventionConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    equality = raw["equality"]
    return InterventionConfig(
        output_dir=Path(raw["output_dir"]),
        seed=int(raw["seed"]),
        frozen_sites={str(model): int(site) for model, site in raw["frozen_sites"].items()},
        equality_templates=tuple(
            (
                EqualityTemplate(
                    id=str(template["id"]),
                    prefix=str(template["prefix"]),
                    suffix=str(template["suffix"]),
                )
                for template in equality["templates"]
            )
        ),
        equality_verbalizers={
            label: tuple((str(value) for value in values))
            for label, values in equality["verbalizers"].items()
        },
        equality_gate=dict(equality["behavior_gate"]),
        quantity_windows=tuple(raw["quantity_windows"]),
        prompt_admission_gate=dict(raw["prompt_admission"]),
        primary_directions=tuple(
            ((str(source), str(target)) for source, target in raw["primary_directions"])
        ),
        subspace=dict(raw["subspace"]),
    )


def adjacent_false_threshold(value: int) -> int:
    if value % 2 == 0:
        return value + 1
    return value - 1


def equality_behavior_trials(
    config: ExperimentConfig, settings: InterventionConfig
) -> list[EqualityBehaviorTrial]:
    trials: list[EqualityBehaviorTrial] = []
    for template, format_name, value in itertools.product(
        settings.equality_templates, config.formats, config.values
    ):
        for variant in variants_for(config, format_name):
            item = surface(config, format_name, value, variant)
            for threshold in (value, adjacent_false_threshold(value)):
                prefix, prompt = template.render(item.expression, threshold)
                expected_true = value == threshold
                trials.append(
                    EqualityBehaviorTrial(
                        trial_id=stable_id(
                            "post-behavior", template.id, format_name, value, variant, threshold
                        ),
                        split="eval",
                        task="equality",
                        template_id=template.id,
                        format=item.format,
                        value=value,
                        variant=variant,
                        threshold=threshold,
                        expected_true=expected_true,
                        prefix=prefix,
                        prompt=prompt,
                    )
                )
    return trials


def _equality_pair_candidates(values: tuple[int, ...], seed: int) -> list[tuple[int, int, int]]:
    candidates = []
    for source, target in itertools.permutations(values, 2):
        candidates.append((source, target, source))
        candidates.append((source, target, target))
    random.Random(seed).shuffle(candidates)
    return candidates


def select_equality_pairs(
    values: tuple[int, ...], *, limit: int, seed: int
) -> list[tuple[int, int, int]]:
    candidates = _equality_pair_candidates(values, seed)
    selected: list[tuple[int, int, int]] = []
    source_counts = {value: 0 for value in values}
    truth_counts = {True: 0, False: 0}
    distance_counts: dict[int, int] = {}
    threshold_direction_counts = {"above": 0, "below": 0}
    while candidates and len(selected) < limit:
        candidates.sort(
            key=lambda row: (
                source_counts[row[0]],
                truth_counts[row[0] == row[2]],
                distance_counts.get(abs(row[0] - row[1]), 0),
                threshold_direction_counts["above" if row[2] > row[0] else "below"],
                stable_id("equality-pair-order", seed, *row),
            )
        )
        row = candidates.pop(0)
        selected.append(row)
        source_counts[row[0]] += 1
        truth_counts[row[0] == row[2]] += 1
        distance = abs(row[0] - row[1])
        distance_counts[distance] = distance_counts.get(distance, 0) + 1
        threshold_direction_counts["above" if row[2] > row[0] else "below"] += 1
    return selected


def _paired_variant(config: ExperimentConfig, format_name: str, replicate: int) -> int:
    options = tuple(variants_for(config, format_name, causal=True))
    return options[replicate % len(options)]


def equality_causal_trials(
    config: ExperimentConfig,
    settings: InterventionConfig,
    *,
    template_id: str,
    control: str = "counterfactual",
    pair_limit: int = 24,
) -> list[EqualityCausalTrial]:
    template = next((item for item in settings.equality_templates if item.id == template_id), None)
    if template is None:
        raise KeyError(template_id)
    if control not in ("counterfactual", "same-number"):
        raise ValueError("control must be 'counterfactual' or 'same-number'")
    trials: list[EqualityCausalTrial] = []
    if control == "counterfactual":
        pairs = select_equality_pairs(
            config.values,
            limit=pair_limit,
            seed=settings.seed + sum((ord(char) for char in template.id)),
        )
    else:
        pairs = [
            (value, value, threshold)
            for value in config.values
            for threshold in (value, adjacent_false_threshold(value))
        ]
    for source_format, target_format in itertools.product(config.formats, repeat=2):
        replicates = max(
            len(tuple(variants_for(config, source_format, causal=True))),
            len(tuple(variants_for(config, target_format, causal=True))),
        )
        for source_value, target_value, threshold in pairs:
            for replicate in range(replicates):
                source_variant = _paired_variant(config, source_format, replicate)
                target_variant = _paired_variant(config, target_format, replicate)
                source_item = surface(config, source_format, source_value, source_variant)
                target_item = surface(config, target_format, target_value, target_variant)
                source_prefix, source_prompt = template.render(source_item.expression, threshold)
                target_prefix, target_prompt = template.render(target_item.expression, threshold)
                source_truth = source_value == threshold
                target_truth = target_value == threshold
                if control == "counterfactual" and source_truth == target_truth:
                    raise AssertionError("Equality counterfactual must flip the answer")
                if control == "same-number" and (
                    source_value != target_value or source_truth != target_truth
                ):
                    raise AssertionError("Equality identity must preserve value and answer")
                trials.append(
                    EqualityCausalTrial(
                        trial_id=stable_id(
                            "post-causal",
                            template.id,
                            control,
                            source_format,
                            target_format,
                            source_value,
                            target_value,
                            source_variant,
                            target_variant,
                            threshold,
                        ),
                        split="eval",
                        task="equality",
                        template_id=template.id,
                        source_format=source_item.format,
                        target_format=target_item.format,
                        source_value=source_value,
                        target_value=target_value,
                        source_variant=source_variant,
                        target_variant=target_variant,
                        threshold=threshold,
                        source_expected_true=source_truth,
                        target_expected_true=target_truth,
                        source_prefix=source_prefix,
                        source_prompt=source_prompt,
                        target_prefix=target_prefix,
                        target_prompt=target_prompt,
                        control=control,
                    )
                )
    return trials
