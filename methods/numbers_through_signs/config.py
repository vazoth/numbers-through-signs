from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    id: str
    short_name: str
    revision: str
    microbatch_size: int
    placement: str = "single"


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    values: tuple[int, ...]
    formats: tuple[str, ...]
    models: tuple[ModelSpec, ...]
    fixed_total: int
    fixed_permutations: int
    run_variants: tuple[dict[str, str], ...]
    class_verbalizers: dict[str, dict[str, tuple[str, ...]]]
    comparison_pair_limit_per_template: int
    parity_pair_limit_per_template: int
    causal_surface_replicates: int


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return ExperimentConfig(
        seed=int(raw["seed"]),
        values=tuple((int(value) for value in raw["values"])),
        formats=tuple((str(value) for value in raw["formats"])),
        models=tuple((ModelSpec(**model) for model in raw["models"])),
        fixed_total=int(raw["fixed_total"]),
        fixed_permutations=int(raw["fixed_permutations"]),
        run_variants=tuple((dict(variant) for variant in raw["run_variants"])),
        class_verbalizers={
            task: {label: tuple(values) for label, values in labels.items()}
            for task, labels in raw["class_verbalizers"].items()
        },
        comparison_pair_limit_per_template=int(raw["comparison_pair_limit_per_template"]),
        parity_pair_limit_per_template=int(raw["parity_pair_limit_per_template"]),
        causal_surface_replicates=int(raw["causal_surface_replicates"]),
    )
