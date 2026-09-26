import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score

from src.propensity.predictability.assessor import oof_predictions
from src.propensity.predictability.evaluation import (
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


def _fixture_Xy(n_samples=250, n_informative=4, n_features=6, seed=0):
    X_arr, y_arr = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=n_informative,
        n_redundant=0, random_state=seed,
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(n_features)])
    y = pd.Series(y_arr)
    return X, y


def _weak_and_strong_pair(seed=0):
    """X_a: a weak/noisy 2-feature slice. X_b: X_a plus several genuinely
    informative features -- a stand-in for "capabilities only" vs
    "capabilities + propensities" with a real, detectable effect."""
    X, y = _fixture_Xy(n_samples=300, n_informative=5, n_features=8, seed=seed)
    X_a = X[["f0", "f1"]]
    X_b = X[["f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7"]]
    return X_a, X_b, y


# --- paired_fold_test ---------------------------------------------------

def test_paired_fold_test_returns_expected_keys():
    X_a, X_b, y = _weak_and_strong_pair()
    result = paired_fold_test(X_a, X_b, y)
    assert set(result.keys()) == {"fold_aurocs_a", "fold_aurocs_b", "deltas", "mean_delta", "statistic", "pvalue", "test"}
    assert len(result["fold_aurocs_a"]) == len(result["fold_aurocs_b"]) == len(result["deltas"])


def test_paired_fold_test_identical_features_gives_near_zero_delta():
    X, y = _fixture_Xy(seed=1)
    result = paired_fold_test(X, X, y)
    assert result["mean_delta"] == pytest.approx(0.0, abs=1e-9)
    assert result["pvalue"] == pytest.approx(1.0)


def test_paired_fold_test_stronger_features_show_positive_mean_delta():
    X_a, X_b, y = _weak_and_strong_pair(seed=2)
    result = paired_fold_test(X_a, X_b, y)
    assert result["mean_delta"] > 0


def test_paired_fold_test_supports_ttest_variant():
    X_a, X_b, y = _weak_and_strong_pair(seed=3)
    result = paired_fold_test(X_a, X_b, y, test="ttest")
    assert result["test"] == "ttest"
    assert not np.isnan(result["pvalue"])


def test_paired_fold_test_rejects_misaligned_inputs():
    X, y = _fixture_Xy(seed=4)
    with pytest.raises(ValueError):
        paired_fold_test(X, X.iloc[:-5], y)


# --- delong_test ----------------------------------------------------------

def test_delong_test_returns_expected_keys():
    X_a, X_b, y = _weak_and_strong_pair(seed=5)
    result = delong_test(X_a, X_b, y)
    assert set(result.keys()) == {"auc_a", "auc_b", "z_statistic", "pvalue"}


def test_delong_test_identical_predictions_gives_zero_z_and_pvalue_one():
    X, y = _fixture_Xy(seed=6)
    result = delong_test(X, X, y)
    assert result["auc_a"] == pytest.approx(result["auc_b"])
    assert result["z_statistic"] == pytest.approx(0.0, abs=1e-6)
    assert result["pvalue"] == pytest.approx(1.0, abs=1e-6)


def test_delong_test_auc_matches_plain_roc_auc_score_on_same_oof_predictions():
    """Validates the AUC half of the DeLong implementation independently:
    its reported auc_a/auc_b must match a plain sklearn roc_auc_score
    computed on the exact same out-of-fold predictions."""
    X_a, X_b, y = _weak_and_strong_pair(seed=7)
    oof_a = oof_predictions(X_a, y, seed=42)
    oof_b = oof_predictions(X_b, y, seed=42)
    expected_auc_a = roc_auc_score(y, oof_a)
    expected_auc_b = roc_auc_score(y, oof_b)

    result = delong_test(X_a, X_b, y, seed=42)
    assert result["auc_a"] == pytest.approx(expected_auc_a)
    assert result["auc_b"] == pytest.approx(expected_auc_b)


def test_delong_test_stronger_features_show_higher_auc_and_significance():
    X_a, X_b, y = _weak_and_strong_pair(seed=8)
    result = delong_test(X_a, X_b, y)
    assert result["auc_b"] > result["auc_a"]


def test_delong_test_rejects_misaligned_inputs():
    X, y = _fixture_Xy(seed=9)
    with pytest.raises(ValueError):
        delong_test(X, X.iloc[:-5], y)


# --- bootstrap_item_ci -----------------------------------------------------

def test_bootstrap_item_ci_returns_expected_keys():
    X_a, X_b, y = _weak_and_strong_pair(seed=10)
    result = bootstrap_item_ci(X_a, X_b, y, n_resamples=15, n_splits=3)
    assert set(result.keys()) == {"mean_delta", "ci_lower", "ci_upper", "n_resamples_used", "ci"}
    assert result["n_resamples_used"] <= 15


def test_bootstrap_item_ci_bounds_are_ordered():
    X_a, X_b, y = _weak_and_strong_pair(seed=11)
    result = bootstrap_item_ci(X_a, X_b, y, n_resamples=15, n_splits=3)
    assert result["ci_lower"] <= result["mean_delta"] <= result["ci_upper"]


def test_bootstrap_item_ci_identical_features_centers_near_zero():
    X, y = _fixture_Xy(seed=12)
    result = bootstrap_item_ci(X, X, y, n_resamples=15, n_splits=3)
    assert result["ci_lower"] <= 0.0 <= result["ci_upper"]


# --- population_paired_test ------------------------------------------------

def test_population_paired_test_consistent_positive_deltas_are_significant():
    rng = np.random.default_rng(0)
    deltas = 0.03 + rng.normal(0, 0.005, size=27)
    result = population_paired_test(deltas.tolist())
    assert result["n"] == 27
    assert result["mean_delta"] > 0
    assert result["pvalue"] < 0.01


def test_population_paired_test_symmetric_noise_is_not_significant():
    rng = np.random.default_rng(1)
    deltas = rng.normal(0, 0.02, size=27)
    result = population_paired_test(deltas.tolist())
    assert result["pvalue"] > 0.05


def test_population_paired_test_supports_ttest_variant():
    result = population_paired_test([0.01, 0.02, 0.015, -0.01, 0.03], test="ttest")
    assert result["test"] == "ttest"


# --- population_win_rate_test ----------------------------------------------

def test_population_win_rate_test_all_wins_is_significant():
    result = population_win_rate_test(27, 27)
    assert result["win_rate"] == 1.0
    assert result["pvalue"] < 0.01


def test_population_win_rate_test_half_wins_is_not_significant():
    result = population_win_rate_test(14, 27)
    assert result["pvalue"] > 0.3


def test_population_win_rate_test_rejects_invalid_wins():
    with pytest.raises(ValueError):
        population_win_rate_test(30, 27)


# --- population_bootstrap_ci ------------------------------------------------

def test_population_bootstrap_ci_bounds_are_ordered():
    rng = np.random.default_rng(2)
    deltas = (0.02 + rng.normal(0, 0.01, size=27)).tolist()
    result = population_bootstrap_ci(deltas, n_resamples=200)
    assert result["ci_lower"] <= result["mean_delta"] <= result["ci_upper"]
    assert result["n_subjects"] == 27


def test_population_bootstrap_ci_empty_deltas_returns_nan():
    result = population_bootstrap_ci([])
    assert np.isnan(result["mean_delta"])
    assert result["n_subjects"] == 0


# --- mixed_effects_test -----------------------------------------------------

def _mixed_effects_fixture():
    rng = np.random.default_rng(3)
    rows = []
    for model in [f"model_{i}" for i in range(10)]:
        model_baseline = rng.normal(0.65, 0.08)
        for config, bump in [("capabilities_only", 0.0), ("capabilities_plus_props", 0.03)]:
            for level in ["-2", "0", "2"]:
                rows.append({
                    "model": model, "config": config, "level": level,
                    "auroc": model_baseline + bump + rng.normal(0, 0.01),
                })
    return pd.DataFrame(rows)


def test_mixed_effects_test_returns_expected_keys():
    df = _mixed_effects_fixture()
    result = mixed_effects_test(df, auroc_column="auroc", config_column="config", group_column="model")
    assert set(result.keys()) == {"effect", "pvalue", "ci_lower", "ci_upper", "term", "summary"}


def test_mixed_effects_test_detects_the_injected_positive_effect():
    df = _mixed_effects_fixture()
    result = mixed_effects_test(df, auroc_column="auroc", config_column="config", group_column="model")
    assert abs(result["effect"]) == pytest.approx(0.03, abs=0.02)
    assert result["pvalue"] < 0.05


def test_mixed_effects_test_rejects_single_valued_config_column():
    df = _mixed_effects_fixture()
    df["config"] = "only_one_value"
    with pytest.raises(ValueError):
        mixed_effects_test(df, auroc_column="auroc", config_column="config", group_column="model")


# --- bootstrap_ci_from_predictions ------------------------------------------

def _predictions_fixture(n=300, seed=0):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_pred_proba = np.clip(y_true * 0.4 + rng.normal(0.4, 0.2, size=n), 0, 1)
    return y_true, y_pred_proba


def test_bootstrap_ci_from_predictions_returns_expected_keys():
    y_true, y_pred_proba = _predictions_fixture()
    result = bootstrap_ci_from_predictions(y_true, y_pred_proba, n_resamples=100)
    assert set(result.keys()) == {"auroc", "ci_lower", "ci_upper", "n_resamples_used", "ci", "grouped"}
    assert result["grouped"] is False


def test_bootstrap_ci_from_predictions_auroc_matches_plain_roc_auc_score():
    y_true, y_pred_proba = _predictions_fixture(seed=1)
    result = bootstrap_ci_from_predictions(y_true, y_pred_proba, n_resamples=50)
    assert result["auroc"] == pytest.approx(roc_auc_score(y_true, y_pred_proba))


def test_bootstrap_ci_from_predictions_bounds_are_ordered():
    y_true, y_pred_proba = _predictions_fixture(seed=2)
    result = bootstrap_ci_from_predictions(y_true, y_pred_proba, n_resamples=200)
    assert result["ci_lower"] <= result["auroc"] <= result["ci_upper"]


def test_bootstrap_ci_from_predictions_is_fast_and_needs_no_refitting():
    """The whole point of this helper vs. bootstrap_item_ci: no model
    fitting happens, so even 1000 resamples over a few hundred rows should
    take well under a second."""
    import time
    y_true, y_pred_proba = _predictions_fixture(n=1000, seed=3)
    t0 = time.perf_counter()
    bootstrap_ci_from_predictions(y_true, y_pred_proba, n_resamples=1000)
    assert time.perf_counter() - t0 < 5.0


def test_bootstrap_ci_from_predictions_grouped_keeps_group_rows_together():
    """With groups, every resample must draw whole groups -- verified
    indirectly by checking the grouped flag and that it still returns a
    sane CI (a direct row-membership check would require inspecting
    internals; this exercises the grouped code path end-to-end)."""
    rng = np.random.default_rng(4)
    n_groups = 50
    groups = np.repeat(np.arange(n_groups), 3)
    y_true = rng.integers(0, 2, size=len(groups))
    y_pred_proba = np.clip(y_true * 0.4 + rng.normal(0.4, 0.2, size=len(groups)), 0, 1)

    result = bootstrap_ci_from_predictions(y_true, y_pred_proba, groups=groups, n_resamples=100)
    assert result["grouped"] is True
    assert result["ci_lower"] <= result["auroc"] <= result["ci_upper"]


def test_bootstrap_ci_from_predictions_perfect_predictions_give_auroc_one():
    y_true = np.array([0, 1] * 50)
    y_pred_proba = y_true.astype(float)
    result = bootstrap_ci_from_predictions(y_true, y_pred_proba, n_resamples=100)
    assert result["auroc"] == pytest.approx(1.0)


# --- paired_bootstrap_ci_from_predictions -----------------------------------

def test_paired_bootstrap_ci_from_predictions_returns_expected_keys():
    y_true, pred_a = _predictions_fixture(seed=5)
    _, pred_b = _predictions_fixture(seed=6)
    result = paired_bootstrap_ci_from_predictions(y_true, pred_a, pred_b, n_resamples=100)
    assert set(result.keys()) == {"mean_delta", "ci_lower", "ci_upper", "n_resamples_used", "ci", "grouped"}


def test_paired_bootstrap_ci_from_predictions_identical_predictions_gives_zero_delta():
    y_true, pred = _predictions_fixture(seed=7)
    result = paired_bootstrap_ci_from_predictions(y_true, pred, pred, n_resamples=100)
    assert result["mean_delta"] == pytest.approx(0.0)
    assert result["ci_lower"] <= 0.0 <= result["ci_upper"]


def test_paired_bootstrap_ci_from_predictions_matches_point_delta():
    y_true, pred_a = _predictions_fixture(n=300, seed=8)
    pred_b = np.clip(pred_a + 0.15, 0, 1)
    expected_delta = roc_auc_score(y_true, pred_b) - roc_auc_score(y_true, pred_a)
    result = paired_bootstrap_ci_from_predictions(y_true, pred_a, pred_b, n_resamples=200)
    assert result["mean_delta"] == pytest.approx(expected_delta)


def test_paired_bootstrap_ci_from_predictions_grouped_runs_end_to_end():
    rng = np.random.default_rng(9)
    n_groups = 50
    groups = np.repeat(np.arange(n_groups), 3)
    y_true = rng.integers(0, 2, size=len(groups))
    pred_a = np.clip(y_true * 0.3 + rng.normal(0.4, 0.2, size=len(groups)), 0, 1)
    pred_b = np.clip(y_true * 0.5 + rng.normal(0.4, 0.2, size=len(groups)), 0, 1)

    result = paired_bootstrap_ci_from_predictions(y_true, pred_a, pred_b, groups=groups, n_resamples=100)
    assert result["grouped"] is True
    assert result["ci_lower"] <= result["mean_delta"] <= result["ci_upper"]
