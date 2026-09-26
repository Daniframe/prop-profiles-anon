"""Side-by-side comparison of named feature-set configurations against the
same label, e.g. "does adding propensity columns to a capability-only
feature set improve an assessor's AUROC?"
"""

from typing import Dict, List, Optional

import pandas as pd

from .assessor import run_ten_fold_cv
from .features import build_feature_table
from . import evaluation

_SIGNIFICANCE_TESTS = {
    "paired_fold": evaluation.paired_fold_test,
    "delong": evaluation.delong_test,
    "bootstrap_ci": evaluation.bootstrap_item_ci,
}


def compare_feature_sets(
    df: pd.DataFrame, feature_sets: Dict[str, List[str]], label_column: str, seed: int = 42,
    significance_tests: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Runs run_ten_fold_cv once per named feature set on the same
    underlying rows/label, returning a DataFrame with one row per set:
    columns "feature_set", "n_features", "n_rows", "auroc". When exactly
    two feature sets are given, an additional "delta_vs_<first_name>"
    column reports each row's AUROC minus the first set's -- the pairwise
    comparison this function exists for, generalized to N named sets since
    that costs nothing extra.

    significance_tests: optional list of names from
    {"paired_fold", "delong", "bootstrap_ci"} (see predictability.evaluation
    for what each does and its cost/rigor tradeoff). Requires exactly two
    feature sets (raises ValueError otherwise) -- these are inherently
    pairwise comparisons. Results are attached to the returned DataFrame's
    .attrs["significance"] dict, keyed by test name, rather than as extra
    columns, since each test returns a differently-shaped result (a CI, a
    p-value + z-statistic, per-fold arrays, etc.) that doesn't fit one row
    per feature set. Population-level tests (population_paired_test,
    population_win_rate_test, population_bootstrap_ci, mixed_effects_test)
    are NOT wired in here -- they operate on deltas/results collected
    across many compare_feature_sets calls (e.g. one per model/incitation
    level), not on a single call's output, so callers assembling those
    should call them directly from predictability.evaluation.
    """
    names = list(feature_sets.keys())
    if significance_tests and len(names) != 2:
        raise ValueError("significance_tests requires exactly two feature sets to compare")

    rows = []
    for name, columns in feature_sets.items():
        X, y = build_feature_table(df, columns, label_column)
        auroc = run_ten_fold_cv(X, y, seed=seed)
        rows.append({"feature_set": name, "n_features": len(columns), "n_rows": len(X), "auroc": auroc})

    result = pd.DataFrame(rows, columns=["feature_set", "n_features", "n_rows", "auroc"])

    if len(names) == 2:
        baseline = result.loc[result["feature_set"] == names[0], "auroc"].iloc[0]
        result[f"delta_vs_{names[0]}"] = result["auroc"] - baseline

        if significance_tests:
            X_a, X_b, y_common = _build_aligned_pair(df, feature_sets[names[0]], feature_sets[names[1]], label_column)
            significance = {}
            for test_name in significance_tests:
                if test_name not in _SIGNIFICANCE_TESTS:
                    raise ValueError(f"Unknown significance test '{test_name}', expected one of {list(_SIGNIFICANCE_TESTS)}")
                significance[test_name] = _SIGNIFICANCE_TESTS[test_name](X_a, X_b, y_common, seed=seed)
            result.attrs["significance"] = significance

    return result


def _build_aligned_pair(df: pd.DataFrame, columns_a: List[str], columns_b: List[str], label_column: str):
    """Builds X_a, X_b, y sharing the exact same row set/order -- the
    intersection of rows surviving both feature sets' NaN-drop, computed by
    dropping NaNs across the union of their columns in one pass -- so
    pairwise significance tests compare like-for-like examples. May drop
    slightly more rows than either feature set's own point-estimate AUROC
    does individually (which drops NaNs independently per set).
    """
    combined_columns = list(dict.fromkeys(columns_a + columns_b))
    X_common, y_common = build_feature_table(df, combined_columns, label_column)
    return X_common[columns_a], X_common[columns_b], y_common
