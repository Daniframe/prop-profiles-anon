"""Statistical-significance tooling for comparing two (or more) predictability
assessor configurations, beyond a single point-estimate AUROC.

Two families of methods here, answering two different questions:

- Per-subject (a single evaluation unit, e.g. one (model, incitation level)
  pair): is one configuration's AUROC improvement over another bigger than
  sampling noise, for THIS subject specifically?
  -> paired_fold_test, delong_test, bootstrap_item_ci.
- Population-level (across many subjects): does the improvement generalise
  -- is it a real effect on average across subjects, not just noise
  concentrated in a few of them?
  -> population_paired_test, population_win_rate_test,
     population_bootstrap_ci, mixed_effects_test.

Plus one standalone utility, bootstrap_ci_from_predictions, for a fast
single-arm AUROC CI on predictions that already exist (no refitting) --
used when the refit-based per-subject/per-resample cost above is too
expensive at scale (see its docstring).

None of these functions load or touch any dataset directly. The per-subject
functions take already-built, ROW-ALIGNED (X_a, X_b, y) -- same examples, same
order -- since that alignment is what makes a "paired" comparison valid;
compare.py's compare_feature_sets is responsible for building that alignment
before calling in. The population-level functions take either a plain list of
per-subject deltas, or (win_rate_test) simple win/total counts, or
(mixed_effects_test) a long-format DataFrame of many subjects' results --
whatever a caller assembles from multiple compare_feature_sets calls.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .assessor import fold_aurocs, oof_predictions

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _assert_aligned(X_a: pd.DataFrame, X_b: pd.DataFrame, y: pd.Series) -> None:
    if not (len(X_a) == len(X_b) == len(y)):
        raise ValueError(
            "X_a, X_b, and y must have the same number of rows, representing the "
            "same underlying examples in the same order -- a paired significance "
            "test is only valid when both configurations are evaluated on "
            "identical examples."
        )


def _paired_test(deltas: np.ndarray, test: str):
    deltas = np.asarray(deltas, dtype=float)
    if test == "wilcoxon":
        if len(deltas) == 0 or np.allclose(deltas, 0):
            return 0.0, 1.0
        statistic, pvalue = stats.wilcoxon(deltas)
    elif test == "ttest":
        if len(deltas) < 2:
            return float("nan"), float("nan")
        statistic, pvalue = stats.ttest_1samp(deltas, popmean=0.0)
    else:
        raise ValueError(f"Unknown test '{test}', expected 'wilcoxon' or 'ttest'")
    return float(statistic), float(pvalue)


# ---------------------------------------------------------------------------
# Per-subject: is THIS subject's improvement real?
# ---------------------------------------------------------------------------


def paired_fold_test(X_a: pd.DataFrame, X_b: pd.DataFrame, y: pd.Series,
                      seed: int = 42, test: str = "wilcoxon") -> Dict:
    """Computes both configurations' per-fold AUROCs on the SAME 10-fold
    KFold split (guaranteed identical partition since KFold depends only on
    row count + seed, not on X's values) and runs a paired significance
    test on the resulting 10 per-fold deltas.

    Cheap -- no extra model fits beyond the normal 10-fold evaluation each
    configuration already needs -- but the 10 folds aren't independent
    samples (their training sets overlap ~90%), so treat the resulting
    p-value as directional/approximate, not an exact hypothesis test.
    """
    _assert_aligned(X_a, X_b, y)
    fold_a = fold_aurocs(X_a, y, seed=seed)
    fold_b = fold_aurocs(X_b, y, seed=seed)

    deltas = np.array(fold_b) - np.array(fold_a)
    statistic, pvalue = _paired_test(deltas, test)

    return {
        "fold_aurocs_a": fold_a,
        "fold_aurocs_b": fold_b,
        "deltas": deltas.tolist(),
        "mean_delta": float(np.mean(deltas)) if len(deltas) else float("nan"),
        "statistic": statistic,
        "pvalue": pvalue,
        "test": test,
    }


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of x, handling ties (average rank within each tie group).
    Building block of the fast DeLong algorithm below."""
    order = np.argsort(x)
    sorted_x = x[order]
    n = len(x)
    ranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    midrank = np.empty(n, dtype=float)
    midrank[order] = ranks
    return midrank


def _fast_delong(predictions_sorted: np.ndarray, m: int):
    """Sun & Xu (2014), "Fast Implementation of DeLong's Algorithm for
    Comparing the Areas Under Correlated Receiver Operating Characteristic
    Curves". predictions_sorted: [n_classifiers, n_examples], columns
    ordered so the m positive examples come first. Returns (aucs, covariance).
    """
    n = predictions_sorted.shape[1] - m
    k = predictions_sorted.shape[0]

    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    for r in range(k):
        tx[r, :] = _compute_midrank(predictions_sorted[r, :m])
        ty[r, :] = _compute_midrank(predictions_sorted[r, m:])
        tz[r, :] = _compute_midrank(predictions_sorted[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    covariance = sx / m + sy / n
    return aucs, covariance


def _delong_roc_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray):
    order = np.argsort(-y_true)
    m = int(np.sum(y_true))
    predictions_sorted = np.vstack((pred_a, pred_b))[:, order]
    aucs, covariance = _fast_delong(predictions_sorted, m)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])
    var = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    if var <= 0:
        return auc_a, auc_b, 0.0, 1.0
    z = (auc_a - auc_b) / np.sqrt(var)
    pvalue = 2 * (1 - stats.norm.cdf(abs(z)))
    return auc_a, auc_b, float(z), float(pvalue)


def delong_test(X_a: pd.DataFrame, X_b: pd.DataFrame, y: pd.Series, seed: int = 42) -> Dict:
    """Builds out-of-fold predicted probabilities for both configurations on
    the SAME 10-fold KFold split, then applies DeLong's test to compare the
    two correlated AUCs computed on those paired, per-example OOF
    predictions -- the standard way to test whether two AUCs measured on
    the same examples differ significantly, with a closed-form p-value (no
    refitting beyond the CV pass itself).

    Note this pools out-of-fold predictions across folds, same as the
    aggregation this module's assessor.py deliberately moved away from for
    the *point-estimate* AUROC -- but that concern doesn't apply here:
    DeLong's test isn't claiming the pooled predictions represent one
    model's generalisation AUROC, it's using them as paired per-example
    observations to test whether two prediction vectors' rankings differ,
    which is exactly what nested-CV-plus-DeLong is standardly used for.
    """
    _assert_aligned(X_a, X_b, y)
    y_true = y.to_numpy()
    oof_a = oof_predictions(X_a, y, seed=seed)
    oof_b = oof_predictions(X_b, y, seed=seed)

    auc_a, auc_b, z, pvalue = _delong_roc_test(y_true, oof_a, oof_b)
    return {"auc_a": auc_a, "auc_b": auc_b, "z_statistic": z, "pvalue": pvalue}


def bootstrap_item_ci(X_a: pd.DataFrame, X_b: pd.DataFrame, y: pd.Series, seed: int = 42,
                       n_resamples: int = 1000, n_splits: int = 10, ci: float = 95) -> Dict:
    """Resamples rows (with replacement) n_resamples times; for each
    resample, refits the full n_splits-fold CV pipeline for both
    configurations (mean-of-folds AUROC, matching assessor.run_ten_fold_cv's
    methodology) and records the AUROC delta. Returns a percentile CI on
    that delta.

    The most expensive but most rigorous option here: each resample
    simulates a new draw from the underlying data-generating population, so
    the resulting CI reflects real sampling variability -- unlike
    paired_fold_test, which only looks at fold-partition variance on one
    fixed dataset. Cost scales as n_resamples x n_splits x 2 model fits;
    reduce n_resamples/n_splits for a quick approximate CI.
    """
    _assert_aligned(X_a, X_b, y)
    rng = np.random.default_rng(seed)
    n = len(y)
    deltas = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        Xa_r = X_a.iloc[idx].reset_index(drop=True)
        Xb_r = X_b.iloc[idx].reset_index(drop=True)
        y_r = y.iloc[idx].reset_index(drop=True)

        aurocs_a = fold_aurocs(Xa_r, y_r, seed=seed, n_splits=n_splits)
        aurocs_b = fold_aurocs(Xb_r, y_r, seed=seed, n_splits=n_splits)
        if not aurocs_a or not aurocs_b:
            continue
        deltas.append(float(np.mean(aurocs_b)) - float(np.mean(aurocs_a)))

    deltas = np.array(deltas)
    lower_pct, upper_pct = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        "mean_delta": float(np.mean(deltas)) if len(deltas) else float("nan"),
        "ci_lower": float(np.percentile(deltas, lower_pct)) if len(deltas) else float("nan"),
        "ci_upper": float(np.percentile(deltas, upper_pct)) if len(deltas) else float("nan"),
        "n_resamples_used": len(deltas),
        "ci": ci,
    }


def paired_bootstrap_ci_from_predictions(y_true, y_pred_proba_a, y_pred_proba_b, groups=None,
                                          seed: int = 42, n_resamples: int = 1000, ci: float = 95) -> Dict:
    """Like bootstrap_ci_from_predictions, but for TWO already-computed
    prediction arrays (from the same y_true, e.g. two configurations' OOF
    predictions on the same rows) -- each resample draws the same indices
    for both arrays and records the AUC delta (b - a), giving a percentile
    CI on the delta without any refitting.

    This is the paired analogue of bootstrap_ci_from_predictions: use this
    (not two separate single-arm CIs eyeballed for overlap) when the actual
    question is "is the delta between two configurations significant" --
    overlapping single-arm CIs do NOT imply a non-significant difference,
    since a paired test accounts for the (typically strong, positive)
    correlation between two configurations' predictions on the same
    examples, which single-arm CIs ignore entirely.
    """
    y_true = np.asarray(y_true)
    pred_a = np.asarray(y_pred_proba_a)
    pred_b = np.asarray(y_pred_proba_b)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []

    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}
        n_groups = len(unique_groups)
        for _ in range(n_resamples):
            drawn_groups = unique_groups[rng.integers(0, n_groups, size=n_groups)]
            idx = np.concatenate([group_to_indices[g] for g in drawn_groups])
            if len(np.unique(y_true[idx])) < 2:
                continue
            deltas.append(roc_auc_score(y_true[idx], pred_b[idx]) - roc_auc_score(y_true[idx], pred_a[idx]))
    else:
        for _ in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            deltas.append(roc_auc_score(y_true[idx], pred_b[idx]) - roc_auc_score(y_true[idx], pred_a[idx]))

    deltas = np.array(deltas)
    lower_pct, upper_pct = (100 - ci) / 2, 100 - (100 - ci) / 2
    point_delta = float("nan")
    if len(np.unique(y_true)) >= 2:
        point_delta = float(roc_auc_score(y_true, pred_b) - roc_auc_score(y_true, pred_a))
    return {
        "mean_delta": point_delta,
        "ci_lower": float(np.percentile(deltas, lower_pct)) if len(deltas) else float("nan"),
        "ci_upper": float(np.percentile(deltas, upper_pct)) if len(deltas) else float("nan"),
        "n_resamples_used": len(deltas),
        "ci": ci,
        "grouped": groups is not None,
    }


def bootstrap_ci_from_predictions(y_true, y_pred_proba, groups=None, seed: int = 42,
                                   n_resamples: int = 1000, ci: float = 95) -> Dict:
    """Single-arm bootstrap CI on the AUROC of an ALREADY-COMPUTED set of
    predictions (e.g. out-of-fold predictions from one CV run, or a fixed
    model's predictions on a held-out test set) -- resamples the (y_true,
    y_pred_proba) pairs with replacement n_resamples times and reports a
    percentile CI on the resulting AUROC distribution.

    This deliberately does NOT refit anything (unlike bootstrap_item_ci,
    which resamples raw rows and refits the full CV pipeline per resample).
    It captures the sampling variability of the AUROC estimator given a
    fixed set of predictions, not the additional variability from retraining
    on different data -- the standard, much cheaper approach for an AUROC CI
    (comparable in spirit to what DeLong's test targets), used here because
    refit-based bootstrapping was empirically measured to take ~30 min for a
    single 355-row/2-config comparison, making it infeasible to run at scale
    (e.g. across many (model, level) subjects or larger populational
    datasets) within any reasonable time budget.

    groups: optional array-like the same length as y_true (e.g. question_id)
    -- when given, resampling draws GROUPS with replacement (all of a drawn
    group's rows included together) rather than independent rows, which is
    the more honest approach when multiple rows share an underlying
    correlated unit (e.g. the same question paired with several models).
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aurocs = []

    if groups is not None:
        groups = np.asarray(groups)
        unique_groups = np.unique(groups)
        group_to_indices = {g: np.where(groups == g)[0] for g in unique_groups}
        n_groups = len(unique_groups)
        for _ in range(n_resamples):
            drawn_groups = unique_groups[rng.integers(0, n_groups, size=n_groups)]
            idx = np.concatenate([group_to_indices[g] for g in drawn_groups])
            if len(np.unique(y_true[idx])) < 2:
                continue
            aurocs.append(roc_auc_score(y_true[idx], y_pred_proba[idx]))
    else:
        for _ in range(n_resamples):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y_true[idx])) < 2:
                continue
            aurocs.append(roc_auc_score(y_true[idx], y_pred_proba[idx]))

    aurocs = np.array(aurocs)
    lower_pct, upper_pct = (100 - ci) / 2, 100 - (100 - ci) / 2
    point_estimate = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) >= 2 else float("nan")
    return {
        "auroc": float(point_estimate),
        "ci_lower": float(np.percentile(aurocs, lower_pct)) if len(aurocs) else float("nan"),
        "ci_upper": float(np.percentile(aurocs, upper_pct)) if len(aurocs) else float("nan"),
        "n_resamples_used": len(aurocs),
        "ci": ci,
        "grouped": groups is not None,
    }


# ---------------------------------------------------------------------------
# Population-level: does the improvement generalise across subjects?
# ---------------------------------------------------------------------------


def population_paired_test(deltas: List[float], test: str = "wilcoxon") -> Dict:
    """Paired significance test (Wilcoxon signed-rank by default, or a
    paired t-test) on a list of subject-level AUROC deltas (e.g. one number
    per (model, level) pair) against the null of zero improvement -- this is
    the test that speaks to generalisation ("is the pattern real across
    subjects, not just noise in a handful of them"), complementing the
    per-subject tests above.
    """
    deltas_arr = np.asarray(deltas, dtype=float)
    statistic, pvalue = _paired_test(deltas_arr, test)
    return {
        "n": len(deltas_arr),
        "mean_delta": float(np.mean(deltas_arr)) if len(deltas_arr) else float("nan"),
        "median_delta": float(np.median(deltas_arr)) if len(deltas_arr) else float("nan"),
        "statistic": statistic,
        "pvalue": pvalue,
        "test": test,
    }


def population_win_rate_test(wins: int, n: int) -> Dict:
    """Binomial (sign) test: is `wins` out of `n` subjects favoring one
    configuration significantly more than the 50% expected by chance?
    Simple and easy to explain, at the cost of discarding effect-size
    information entirely -- a companion to, not a replacement for,
    population_paired_test.
    """
    if not (0 <= wins <= n):
        raise ValueError(f"wins ({wins}) must be between 0 and n ({n})")
    result = stats.binomtest(wins, n, p=0.5, alternative="two-sided")
    return {"wins": wins, "n": n, "win_rate": wins / n if n else float("nan"), "pvalue": float(result.pvalue)}


def population_bootstrap_ci(deltas: List[float], seed: int = 42,
                             n_resamples: int = 1000, ci: float = 95) -> Dict:
    """Bootstrap-resamples the list of subject-level deltas (not the
    underlying items) to get a CI on the population-level mean improvement.
    Cheap -- reuses already-computed per-subject deltas, no refitting.
    """
    deltas_arr = np.asarray(deltas, dtype=float)
    n = len(deltas_arr)
    if n == 0:
        return {"mean_delta": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan"),
                "n_subjects": 0, "n_resamples": n_resamples, "ci": ci}

    rng = np.random.default_rng(seed)
    means = np.array([deltas_arr[rng.integers(0, n, size=n)].mean() for _ in range(n_resamples)])
    lower_pct, upper_pct = (100 - ci) / 2, 100 - (100 - ci) / 2
    return {
        "mean_delta": float(deltas_arr.mean()),
        "ci_lower": float(np.percentile(means, lower_pct)),
        "ci_upper": float(np.percentile(means, upper_pct)),
        "n_subjects": n,
        "n_resamples": n_resamples,
        "ci": ci,
    }


def mixed_effects_test(df: pd.DataFrame, auroc_column: str, config_column: str, group_column: str) -> Dict:
    """Fits a random-intercept mixed-effects model (auroc_column ~
    config_column, random intercept per group_column, e.g. per model) via
    statsmodels, estimating the configuration effect while accounting for
    repeated measures (e.g. the same model appearing at multiple incitation
    levels) -- the most rigorous option here, at the cost of needing more
    data and a normality-of-residuals assumption the nonparametric
    alternatives above don't.
    """
    import statsmodels.formula.api as smf

    work = df[[auroc_column, config_column, group_column]].copy()
    work.columns = ["auroc", "config", "group"]

    fit = smf.mixedlm("auroc ~ config", work, groups=work["group"]).fit()

    config_terms = [name for name in fit.params.index if name.startswith("config")]
    if not config_terms:
        raise ValueError(f"'{config_column}' must have at least two distinct values")
    term = config_terms[0]
    conf_int = fit.conf_int().loc[term]

    return {
        "effect": float(fit.params[term]),
        "pvalue": float(fit.pvalues[term]),
        "ci_lower": float(conf_int[0]),
        "ci_upper": float(conf_int[1]),
        "term": term,
        "summary": str(fit.summary()),
    }
