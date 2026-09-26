"""Sector 1: synthetic/external benchmark generation (paper stage 1).

- spec: BenchmarkSpec dataclass + benchmarks/*.yaml loader.
- engine: templated generation (risk_levels for RA, color_permutation for BR).
- context_generation: Azure structured-output seed generation for context-grounded
  benchmarks (Ex; Ul's seeds are pre-generated and checked in, see module docstring).
- render: seed -> rendered multiple choice question (Ex, Ul).
- external: adapter for real/external benchmarks needing no generation (TimeMenatQA).
"""

from .spec import BenchmarkSpec, load_spec
from .engine import generate_risk_levels, generate_color_permutation
from .context_generation import generate_context_seeds
from .render import render_seed_questions
from .external import load_external_benchmark

__all__ = [
    "BenchmarkSpec",
    "load_spec",
    "generate_risk_levels",
    "generate_color_permutation",
    "generate_context_seeds",
    "render_seed_questions",
    "load_external_benchmark",
]
