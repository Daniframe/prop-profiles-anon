"""Sector 5: predictability -- does a feature set (capability demands,
propensity demands, or any other item-level columns) help predict
per-instance model correctness, and does adding one feature set to another
improve on it, with statistical rigor behind that claim?

- features: build_feature_table, a generic column-selection + NaN-drop step.
- assessor: run_ten_fold_cv (+ fold_aurocs/oof_predictions), the one point-
  estimate evaluation method implemented so far.
- compare: compare_feature_sets, side-by-side AUROC across named feature
  sets, with optional pairwise significance testing.
- evaluation: per-subject (paired_fold_test, delong_test, bootstrap_item_ci)
  and population-level (population_paired_test, population_win_rate_test,
  population_bootstrap_ci, mixed_effects_test) significance tests.
"""

from .features import build_feature_table
from .assessor import run_ten_fold_cv, fold_aurocs, oof_predictions
from .compare import compare_feature_sets
from .evaluation import (
    paired_fold_test,
    delong_test,
    bootstrap_item_ci,
    bootstrap_ci_from_predictions,
    paired_bootstrap_ci_from_predictions,
    population_paired_test,
    population_win_rate_test,
    population_bootstrap_ci,
    mixed_effects_test,
)

__all__ = [
    "build_feature_table",
    "run_ten_fold_cv",
    "fold_aurocs",
    "oof_predictions",
    "compare_feature_sets",
    "paired_fold_test",
    "delong_test",
    "bootstrap_item_ci",
    "bootstrap_ci_from_predictions",
    "paired_bootstrap_ci_from_predictions",
    "population_paired_test",
    "population_win_rate_test",
    "population_bootstrap_ci",
    "mixed_effects_test",
]
