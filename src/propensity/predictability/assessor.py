"""Random Forest assessor evaluation: predicts a binary outcome (e.g. model
correctness) from an arbitrary feature matrix via 10-fold cross-validation.

RandomForestClassifier hyperparameters (min_samples_split=50 + entropy
criterion) match the original assessors.py's
run_ten_fold_ID_evaluation. AUROC aggregation deliberately does NOT match
that original code, though: the original pools every fold's held-out
predictions into one array and computes a single roc_auc_score over the
pool. That treats predictions from 10 different fitted models as directly
comparable rankings, which they aren't -- each fold fits its own Random
Forest on different training data, so the resulting score isn't the
generalisation performance of one model. The methodologically sound
aggregation is the mean of each fold's own AUROC (computed against only
that fold's model and held-out data), which is what run_ten_fold_cv does.

fold_aurocs and oof_predictions are exposed separately (not just inlined
into run_ten_fold_cv) because predictability.evaluation's significance
tests need the same per-fold split for two different feature sets at once:
sklearn's KFold(shuffle=True, random_state=seed).split(X) depends only on
len(X) and the seed, never on X's values, so calling these with the same
seed on two equal-length feature sets deterministically reproduces the
same row partition -- giving paired folds/predictions "for free" without
either function needing to know a comparison is happening.
"""

from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def fold_aurocs(X: pd.DataFrame, y: pd.Series, seed: int = 42, n_splits: int = 10) -> List[float]:
    """Fits a fresh RandomForestClassifier per fold and returns one AUROC
    per fold (that fold's held-out predictions against that fold's model
    only). A fold whose held-out set contains only one class is skipped
    (AUROC is undefined there) rather than counted as an error.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs = []

    for train_idx, test_idx in kf.split(X):
        y_test = y.iloc[test_idx]
        if y_test.nunique() < 2:
            continue
        model = RandomForestClassifier(random_state=seed, min_samples_split=50, criterion="entropy")
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred_proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        aurocs.append(roc_auc_score(y_test, y_pred_proba))

    return aurocs


def oof_predictions(X: pd.DataFrame, y: pd.Series, seed: int = 42, n_splits: int = 10) -> np.ndarray:
    """Returns one out-of-fold predicted probability per row (from
    whichever fold's model held that row out) -- unlike fold_aurocs, no
    rows are skipped regardless of a fold's class balance, since generating
    a prediction doesn't require computing that fold's own AUROC. Useful
    for methods that need paired per-example predictions across two
    configurations (e.g. DeLong's test) rather than a single scalar.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.full(len(y), np.nan)

    for train_idx, test_idx in kf.split(X):
        model = RandomForestClassifier(random_state=seed, min_samples_split=50, criterion="entropy")
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds[test_idx] = model.predict_proba(X.iloc[test_idx])[:, 1]

    return preds


def run_ten_fold_cv(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> float:
    """Mean of fold_aurocs -- the point-estimate AUROC for one feature set.
    Returns NaN if every fold gets skipped (e.g. y has only one class overall).
    """
    aurocs = fold_aurocs(X, y, seed=seed)
    return float(np.mean(aurocs)) if aurocs else float("nan")
