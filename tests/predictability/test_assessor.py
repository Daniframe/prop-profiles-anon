import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

from src.propensity.predictability.assessor import run_ten_fold_cv


def _fixture_Xy(n_samples=200, n_informative=4, seed=0):
    X_arr, y_arr = make_classification(
        n_samples=n_samples, n_features=6, n_informative=n_informative,
        n_redundant=0, random_state=seed,
    )
    X = pd.DataFrame(X_arr, columns=[f"f{i}" for i in range(X_arr.shape[1])])
    y = pd.Series(y_arr)
    return X, y


def test_run_ten_fold_cv_returns_auroc_in_valid_range():
    X, y = _fixture_Xy()
    auroc = run_ten_fold_cv(X, y)
    assert 0.0 <= auroc <= 1.0


def test_run_ten_fold_cv_is_deterministic_for_same_seed():
    X, y = _fixture_Xy()
    first = run_ten_fold_cv(X, y, seed=42)
    second = run_ten_fold_cv(X, y, seed=42)
    assert first == second


def test_run_ten_fold_cv_informative_features_beat_pure_noise():
    """A dataset with genuinely predictive features should score well above
    chance (0.5); this is a sanity check the pipeline isn't silently
    scrambling labels/predictions rather than a tight numeric assertion."""
    X, y = _fixture_Xy(n_samples=300, n_informative=5, seed=1)
    auroc = run_ten_fold_cv(X, y)
    assert auroc > 0.7


def test_run_ten_fold_cv_pure_noise_is_near_chance():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(300, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(rng.integers(0, 2, size=300))
    auroc = run_ten_fold_cv(X, y)
    assert 0.3 < auroc < 0.7


def test_run_ten_fold_cv_averages_per_fold_aurocs_not_pooled():
    """Regression test: AUROC must be the mean of each fold's own AUROC
    (computed against only that fold's model), never a single AUROC over
    predictions pooled across folds -- pooling mixes rankings from 10
    different fitted models, which isn't methodologically valid. Verified
    by independently replicating the fold loop here and comparing."""
    X, y = _fixture_Xy(n_samples=250, n_informative=4, seed=5)
    seed = 42

    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    expected_fold_aurocs = []
    for train_idx, test_idx in kf.split(X):
        model = RandomForestClassifier(random_state=seed, min_samples_split=50, criterion="entropy")
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        expected_fold_aurocs.append(roc_auc_score(y.iloc[test_idx], y_pred_proba))
    expected = np.mean(expected_fold_aurocs)

    actual = run_ten_fold_cv(X, y, seed=seed)
    assert actual == pytest.approx(expected, rel=1e-9)


def test_run_ten_fold_cv_skips_single_class_folds_without_crashing():
    """With extreme class imbalance, some 10-fold splits may land a fold
    whose held-out set is entirely one class (AUROC undefined there); this
    must be skipped rather than raising, and still return a valid score."""
    rng = np.random.default_rng(9)
    n = 100
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series(np.concatenate([np.ones(96), np.zeros(4)]).astype(int))
    auroc = run_ten_fold_cv(X, y)
    assert 0.0 <= auroc <= 1.0


def test_run_ten_fold_cv_returns_nan_when_label_has_only_one_class():
    X = pd.DataFrame(np.random.default_rng(1).normal(size=(50, 3)), columns=["a", "b", "c"])
    y = pd.Series(np.ones(50, dtype=int))
    auroc = run_ten_fold_cv(X, y)
    assert np.isnan(auroc)
