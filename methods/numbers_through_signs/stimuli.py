from __future__ import annotations
import hashlib
import itertools
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import Literal
from .config import ExperimentConfig

Split = Literal["dev", "eval"]
Task = Literal["comparison", "parity"]
Format = Literal["digit", "word", "run", "fixed"]
CARRIER = " ;\n"
NUMBER_WORDS = {
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}


@dataclass(frozen=True)
class Template:
    id: str
    split: Split
    task: Task
    prefix: str
    suffix: str
    parity_property: Literal["even"] | None = None

    def render(self, expression: str, threshold: int | None = None) -> tuple[str, str]:
        prefix = self.prefix.format(expression=expression)
        if not prefix.endswith(CARRIER):
            raise ValueError(
                f"Template {self.id} must end its quantity prefix with the shared carrier"
            )
        suffix = self.suffix.format(threshold=threshold)
        return (prefix, prefix + suffix)


TEMPLATES: tuple[Template, ...] = (
    Template(
        "cmp_dev_value",
        "dev",
        "comparison",
        "Value: {expression} ;\n",
        "Compared with {threshold}, this value is",
    ),
    Template(
        "cmp_dev_quantity",
        "dev",
        "comparison",
        "Quantity: {expression} ;\n",
        "Relative to {threshold}, this quantity is",
    ),
    Template(
        "cmp_dev_given",
        "dev",
        "comparison",
        "Given quantity: {expression} ;\n",
        "With {threshold} as reference, this quantity is",
    ),
    Template(
        "cmp_eval_number",
        "eval",
        "comparison",
        "Number: {expression} ;\n",
        "In comparison with {threshold}, this number is",
    ),
    Template(
        "cmp_eval_represented",
        "eval",
        "comparison",
        "Quantity represented: {expression} ;\n",
        "Against {threshold}, the represented quantity is",
    ),
    Template(
        "cmp_eval_cardinality",
        "eval",
        "comparison",
        "Cardinality: {expression} ;\n",
        "Taking {threshold} as the comparator, this cardinality is",
    ),
    Template(
        "par_eval_number",
        "eval",
        "parity",
        "Number: {expression} ;\n",
        "Classified by parity, this integer is",
        "even",
    ),
    Template(
        "par_eval_represented",
        "eval",
        "parity",
        "Quantity represented: {expression} ;\n",
        "As an integer, this represented quantity is",
        "even",
    ),
    Template(
        "par_eval_cardinality",
        "eval",
        "parity",
        "Cardinality: {expression} ;\n",
        "The parity class of this cardinality is",
        "even",
    ),
)


@dataclass(frozen=True)
class Surface:
    format: Format
    value: int
    variant: int
    expression: str


@dataclass(frozen=True)
class BehaviorTrial:
    trial_id: str
    split: Split
    task: Task
    template_id: str
    format: Format
    value: int
    variant: int
    threshold: int | None
    expected_true: bool
    prefix: str
    prompt: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CausalTrial:
    trial_id: str
    split: Split
    task: Task
    template_id: str
    source_format: Format
    target_format: Format
    source_value: int
    target_value: int
    source_variant: int
    target_variant: int
    threshold: int | None
    source_expected_true: bool
    target_expected_true: bool
    source_prefix: str
    source_prompt: str
    target_prefix: str
    target_prompt: str
    control: str = "counterfactual"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def stable_id(*parts: object) -> str:
    text = "|".join((str(part) for part in parts))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def expected_truth(template: Template, value: int, threshold: int | None) -> bool:
    if template.task == "comparison":
        if threshold is None:
            raise ValueError("Comparison requires a threshold")
        return value > threshold
    if template.parity_property == "even":
        return value % 2 == 0
    raise ValueError(f"Missing parity property for {template.id}")


def _fixed_tokens(value: int, total: int, seed: int, variant: int) -> str:
    if not 0 <= value <= total:
        raise ValueError(f"Cannot place {value} target tokens in a list of length {total}")
    tokens = ["A"] * value + ["B"] * (total - value)
    rng = random.Random(seed + value * 1009 + variant * 9176)
    rng.shuffle(tokens)
    return " ".join(tokens)


def surface(config: ExperimentConfig, format_name: str, value: int, variant: int = 0) -> Surface:
    if format_name == "digit":
        expression = str(value)
        variant = 0
    elif format_name == "word":
        expression = NUMBER_WORDS[value]
        variant = 0
    elif format_name == "run":
        spec = config.run_variants[variant % len(config.run_variants)]
        sequence = spec["separator"].join([spec["item"]] * value)
        expression = f"the number of {spec['item']} tokens in [{sequence}]"
    elif format_name == "fixed":
        sequence = _fixed_tokens(value, config.fixed_total, config.seed, variant)
        expression = f"the number of A tokens in [{sequence}]"
    else:
        raise KeyError(f"Unknown format {format_name!r}")
    return Surface(format_name, value, variant, expression)


def variants_for(config: ExperimentConfig, format_name: str, causal: bool = False) -> range:
    if format_name in ("digit", "word"):
        return range(1)
    if format_name == "run":
        count = len(config.run_variants)
    elif format_name == "fixed":
        count = config.fixed_permutations
    else:
        raise KeyError(format_name)
    if causal:
        count = min(count, config.causal_surface_replicates)
    return range(count)


def behavior_trials(config: ExperimentConfig) -> Iterator[BehaviorTrial]:
    thresholds = tuple(range(min(config.values) + 1, max(config.values)))
    for template in TEMPLATES:
        task_thresholds: Iterable[int | None]
        task_thresholds = thresholds if template.task == "comparison" else (None,)
        for format_name, value, threshold in itertools.product(
            config.formats, config.values, task_thresholds
        ):
            for actual_variant in variants_for(config, format_name):
                item = surface(config, format_name, value, actual_variant)
                prefix, prompt = template.render(item.expression, threshold)
                truth = expected_truth(template, value, threshold)
                trial_id = stable_id(
                    "behavior", template.id, format_name, value, actual_variant, threshold
                )
                yield BehaviorTrial(
                    trial_id,
                    template.split,
                    template.task,
                    template.id,
                    item.format,
                    value,
                    actual_variant,
                    threshold,
                    truth,
                    prefix,
                    prompt,
                )


def _comparison_pairs(values: tuple[int, ...], limit: int, seed: int) -> list[tuple[int, int, int]]:
    pairs: list[tuple[int, int, int]] = []
    for source, target in itertools.permutations(values, 2):
        between = [
            threshold
            for threshold in range(min(values) + 1, max(values))
            if (source > threshold) != (target > threshold)
        ]
        if between:
            threshold = between[(source + target + seed) % len(between)]
            pairs.append((source, target, threshold))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if limit <= 0 or limit >= len(pairs):
        return pairs
    selected: list[tuple[int, int, int]] = []
    source_counts = {value: 0 for value in values}
    direction_counts = {True: 0, False: 0}
    while pairs and len(selected) < limit:
        pairs.sort(
            key=lambda row: (
                source_counts[row[0]],
                direction_counts[row[0] > row[1]],
                abs(row[0] - row[1]),
            )
        )
        row = pairs.pop(0)
        selected.append(row)
        source_counts[row[0]] += 1
        direction_counts[row[0] > row[1]] += 1
    return selected


def _parity_pairs(values: tuple[int, ...], limit: int, seed: int) -> list[tuple[int, int, None]]:
    pairs = [
        (source, target, None)
        for source, target in itertools.permutations(values, 2)
        if source % 2 != target % 2
    ]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if limit <= 0 or limit >= len(pairs):
        return pairs
    selected: list[tuple[int, int, None]] = []
    source_counts = {value: 0 for value in values}
    while pairs and len(selected) < limit:
        pairs.sort(key=lambda row: (source_counts[row[0]], abs(row[0] - row[1])))
        row = pairs.pop(0)
        selected.append(row)
        source_counts[row[0]] += 1
    return selected


def _paired_variant(config: ExperimentConfig, format_name: str, replicate: int) -> int:
    options = tuple(variants_for(config, format_name, causal=True))
    return options[replicate % len(options)]


def causal_trials(config: ExperimentConfig) -> Iterator[CausalTrial]:
    for template in TEMPLATES:
        if template.task == "comparison":
            pairs = _comparison_pairs(
                config.values,
                config.comparison_pair_limit_per_template,
                config.seed + sum((ord(char) for char in template.id)),
            )
        else:
            pairs = _parity_pairs(
                config.values,
                config.parity_pair_limit_per_template,
                config.seed + sum((ord(char) for char in template.id)),
            )
        for source_format, target_format in itertools.product(config.formats, repeat=2):
            replicates = max(
                len(tuple(variants_for(config, source_format, causal=True))),
                len(tuple(variants_for(config, target_format, causal=True))),
            )
            for source_value, target_value, threshold in pairs:
                for replicate in range(replicates):
                    source_variant = _paired_variant(config, source_format, replicate)
                    target_variant = _paired_variant(config, target_format, replicate)
                    source_surface = surface(config, source_format, source_value, source_variant)
                    target_surface = surface(config, target_format, target_value, target_variant)
                    source_prefix, source_prompt = template.render(
                        source_surface.expression, threshold
                    )
                    target_prefix, target_prompt = template.render(
                        target_surface.expression, threshold
                    )
                    source_truth = expected_truth(template, source_value, threshold)
                    target_truth = expected_truth(template, target_value, threshold)
                    if source_truth == target_truth:
                        raise AssertionError("Counterfactual trial must flip the high-level answer")
                    trial_id = stable_id(
                        "causal",
                        template.id,
                        source_format,
                        target_format,
                        source_value,
                        target_value,
                        source_variant,
                        target_variant,
                        threshold,
                    )
                    yield CausalTrial(
                        trial_id,
                        template.split,
                        template.task,
                        template.id,
                        source_surface.format,
                        target_surface.format,
                        source_value,
                        target_value,
                        source_variant,
                        target_variant,
                        threshold,
                        source_truth,
                        target_truth,
                        source_prefix,
                        source_prompt,
                        target_prefix,
                        target_prompt,
                    )


def identity_trials(config: ExperimentConfig) -> Iterator[CausalTrial]:
    """Same-cardinality trials for preservation controls."""
    comparison_threshold = (min(config.values) + max(config.values)) // 2
    for template in TEMPLATES:
        threshold = comparison_threshold if template.task == "comparison" else None
        for source_format, target_format in itertools.product(config.formats, repeat=2):
            replicates = max(
                len(tuple(variants_for(config, source_format, causal=True))),
                len(tuple(variants_for(config, target_format, causal=True))),
            )
            for value in config.values:
                for replicate in range(replicates):
                    source_variant = _paired_variant(config, source_format, replicate)
                    target_variant = _paired_variant(config, target_format, replicate)
                    source_surface = surface(config, source_format, value, source_variant)
                    target_surface = surface(config, target_format, value, target_variant)
                    source_prefix, source_prompt = template.render(
                        source_surface.expression, threshold
                    )
                    target_prefix, target_prompt = template.render(
                        target_surface.expression, threshold
                    )
                    truth = expected_truth(template, value, threshold)
                    trial_id = stable_id(
                        "identity",
                        template.id,
                        source_format,
                        target_format,
                        value,
                        source_variant,
                        target_variant,
                        threshold,
                    )
                    yield CausalTrial(
                        trial_id,
                        template.split,
                        template.task,
                        template.id,
                        source_surface.format,
                        target_surface.format,
                        value,
                        value,
                        source_variant,
                        target_variant,
                        threshold,
                        truth,
                        truth,
                        source_prefix,
                        source_prompt,
                        target_prefix,
                        target_prompt,
                        "same-number",
                    )
