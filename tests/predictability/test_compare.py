import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.propensity.predictability.compare import compare_feature_sets


def _fixture_df(n_samples=200, seed=0):
    X_arr, y_arr = make_classification(
        n_samples=n_samples, n_features=6, n_informative=4, n_redundant=0, random_state=seed,
    )
    df = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
    df["label"] = y_arr
    return df


def test_compare_feature_sets_returns_one_row_per_named_set():
    df = _fixture_df()
    feature_sets = {"baseline": ["f0", "f1"], "augmented": ["f0", "f1", "f2", "f3"]}
    result = compare_feature_sets(df, feature_sets, "label")
    assert set(result["feature_set"]) == {"baseline", "augmented"}
    assert len(result) == 2


def test_compare_feature_sets_reports_correct_feature_counts():
    df = _fixture_df()
    feature_sets = {"baseline": ["f0", "f1"], "augmented": ["f0", "f1", "f2", "f3"]}
    result = compare_feature_sets(df, feature_sets, "label")
    counts = dict(zip(result["feature_set"], result["n_features"]))
    assert counts["baseline"] == 2
    assert counts["augmented"] == 4


def test_compare_feature_sets_adds_delta_column_for_two_sets():
    df = _fixture_df()
    feature_sets = {"baseline": ["f0", "f1"], "augmented": ["f0", "f1", "f2", "f3"]}
    result = compare_feature_sets(df, feature_sets, "label")
    assert "delta_vs_baseline" in result.columns
    baseline_auroc = result.loc[result["feature_set"] == "baseline", "auroc"].iloc[0]
    augmented_row = result.loc[result["feature_set"] == "augmented"].iloc[0]
    assert augmented_row["delta_vs_baseline"] == augmented_row["auroc"] - baseline_auroc


def test_compare_feature_sets_no_delta_column_for_three_sets():
    df = _fixture_df()
    feature_sets = {"a": ["f0"], "b": ["f0", "f1"], "c": ["f0", "f1", "f2"]}
    result = compare_feature_sets(df, feature_sets, "label")
    assert not any(col.startswith("delta_vs_") for col in result.columns)


def test_compare_feature_sets_more_informative_features_score_at_least_as_well():
    """Not a strict guarantee for any single random draw, but with a large,
    clearly-informative feature added, augmented should not do meaningfully
    worse than baseline on average -- a loose sanity check of the wiring."""
    df = _fixture_df(n_samples=400, seed=7)
    feature_sets = {"baseline": ["f0"], "augmented": ["f0", "f1", "f2", "f3", "f4", "f5"]}
    result = compare_feature_sets(df, feature_sets, "label")
    baseline_auroc = result.loc[result["feature_set"] == "baseline", "auroc"].iloc[0]
    augmented_auroc = result.loc[result["feature_set"] == "augmented", "auroc"].iloc[0]
    assert augmented_auroc > baseline_auroc - 0.05


def test_compare_feature_sets_significance_tests_requires_exactly_two_sets():
    df = _fixture_df()
    feature_sets = {"a": ["f0"], "b": ["f0", "f1"], "c": ["f0", "f1", "f2"]}
    with pytest.raises(ValueError):
        compare_feature_sets(df, feature_sets, "label", significance_tests=["paired_fold"])


def test_compare_feature_sets_rejects_unknown_significance_test_name():
    df = _fixture_df()
    feature_sets = {"baseline": ["f0", "f1"], "augmented": ["f0", "f1", "f2", "f3"]}
    with pytest.raises(ValueError):
        compare_feature_sets(df, feature_sets, "label", significance_tests=["not_a_real_test"])


def test_compare_feature_sets_attaches_requested_significance_results():
    df = _fixture_df(n_samples=300, seed=9)
    feature_sets = {"baseline": ["f0", "f1"], "augmented": ["f0", "f1", "f2", "f3", "f4", "f5"]}
    result = compare_feature_sets(df, feature_sets, "label", significance_tests=["paired_fold", "delong"])
    assert "significance" in result.attrs
    assert set(result.attrs["significance"].keys()) == {"paired_fold", "delong"}
    assert "pvalue" in result.attrs["significance"]["paired_fold"]
    assert "pvalue" in result.attrs["significance"]["delong"]


def test_compare_feature_sets_no_significance_attr_when_not_requested():
    df = _fixture_df()
    feature_sets = {"baseline": ["f0", "f1"], "augmented": ["f0", "f1", "f2", "f3"]}
    result = compare_feature_sets(df, feature_sets, "label")
    assert "significance" not in result.attrs
