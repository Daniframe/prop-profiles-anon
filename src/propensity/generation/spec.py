"""Declarative benchmark specs (benchmarks/*.yaml) driving Sector 1 generation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Union

import yaml

VALID_KINDS = {"risk_levels", "color_permutation", "context_grounded", "external"}


@dataclass
class BenchmarkSpec:
    """A benchmark's generation recipe.

    name: the benchmark/dimension short code (e.g. "RA", "BR", "Ex", "Ul", "TimeMenatQA").
    kind: which generation engine to dispatch to -- one of VALID_KINDS.
    params: kind-specific parameters, passed through as-is from the YAML file
        (e.g. levels_to_ev_ratios for risk_levels, template/color/option file
        paths for color_permutation, contexts_file/rubric_path/level_pairs for
        context_grounded, source_path/column_map for external).
    """

    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f"Unknown benchmark kind '{self.kind}' (expected one of {sorted(VALID_KINDS)})")


def load_spec(path: Union[str, Path]) -> BenchmarkSpec:
    """Loads a BenchmarkSpec from a benchmarks/*.yaml file."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    name = raw.pop("name")
    kind = raw.pop("kind")
    return BenchmarkSpec(name=name, kind=kind, params=raw)
