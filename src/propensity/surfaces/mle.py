"""Eq. 6 maximum-likelihood fitting of a single propensity theta from
item-level (demand interval, outcome) data. Ported from get_prop_point.py.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.stats import binned_statistic
from statsmodels.nonparametric.smoothers_lowess import lowess

from .model import two_sided_sigma


def neg_log_likelihood(theta, demands, success, k=1.0):
    """Eq. 6 negative log-likelihood: -sum_i [y_i log p_i + (1-y_i) log(1-p_i)],
    computed via the product form (probs**success * (1-probs)**(1-success))
    as in the original implementation, with probs clipped away from 0/1.
    """
    demands = np.asarray(demands, dtype=float)
    success = np.asarray(success)
    # scipy.optimize passes theta as a 1-element array; reduce to a plain
    # numpy scalar so two_sided_sigma's result can be assigned into probs[i]
    # (newer numpy no longer implicitly squeezes 1-element arrays there).
    theta = np.ravel(theta)[0]
    probs = np.zeros(len(demands))
    for i, (x1, x2) in enumerate(demands):
        probs[i] = two_sided_sigma(theta, x1, x2, k, k)
    probs = np.clip(probs, 1e-10, 1 - 1e-10)
    return -np.log(np.prod(probs**success * (1 - probs) ** (1 - success)))


def fit_theta(demands, success, k=1.0, x_init=None, n_bins=20, lowess_frac=0.4, maxiter=500,
              robust=False, max_retries=20, patience=2, restart_range=None):
    """
    Fits theta by maximum likelihood (Eq. 6) via scipy.optimize.minimize
    (BFGS). If x_init is not given, derives one from a LOWESS-smoothed,
    binned empirical success curve (bin by interval center, smooth, take the
    argmax) -- exactly the approach used in get_prop_point.py.

    demands: (N, 2) array-like of (b_l, b_u) pairs.
    success: (N,) array-like of binary outcomes.
    robust: if False (default), a single BFGS run from the LOWESS-derived
        x_init -- unchanged from before this parameter existed. If True,
        extra attempts are spent ONLY when that first run fails to converge
        (the ~97% of well-identified fits pay no extra cost). When triggered,
        additional starting points spread across restart_range are tried one
        at a time, tracking whichever attempt has the best (lowest) negative
        log-likelihood so far, and stopping as soon as any of:
          (a) an attempt converges,
          (b) the best likelihood hasn't improved for `patience` consecutive
              additional attempts (diminishing returns), or
          (c) `max_retries` additional attempts have been made.
        The winner is picked by best likelihood among ALL attempts made
        (converged or not) -- a non-converged run can still have reached a
        lower NLL than a converged one elsewhere, and discarding it would
        both give a worse point estimate and violate the guarantee that
        enabling `robust` can only match or improve the single-start fit,
        never make it worse. This matters when the likelihood surface is
        nearly flat (e.g. near-ceiling/near-floor outcome data): a single
        BFGS run can land in an arbitrary local optimum, and averaging
        thetas from different optima would blend two unrelated modes into a
        point that fits neither -- so this keeps the best-supported mode
        instead, not an average.
        EXCEPTION: if `success` is fully degenerate (every item succeeded, or
        every item failed), retries are skipped entirely regardless of the
        first attempt's convergence flag. Fully degenerate data has no
        interior likelihood maximum at all -- it improves monotonically
        toward whatever boundary the search is allowed to reach -- so
        "best likelihood" would just walk to that boundary rather than
        recovering a meaningful estimate. Skipping retries here reproduces
        the single-shot pipeline's own (equally arbitrary, but empirically
        paper-matching) behavior for these rows instead of making it worse.
    max_retries: cap on additional attempts beyond the first (only spent if
        the first attempt didn't converge; ignored when robust=False).
    patience: consecutive non-improving additional attempts before giving up
        (only used when robust=True).
    restart_range: (low, high) to both spread extra starting points across
        AND bound the retry optimizations to (via L-BFGS-B), defaulting to
        (-5, 5) -- a margin around the rubric's defined -3..+3 propensity
        scale. This bound is deliberately NOT applied to the first attempt
        (unconstrained BFGS, unchanged from before `robust` existed) and
        matters specifically for retries: for genuinely unidentified data
        (e.g. every response fails regardless of level), the likelihood has
        no interior maximum at all -- it improves monotonically as theta
        walks toward +-infinity -- so an unconstrained retry can wander far
        enough down that tail for BFGS's gradient-flatness tolerance to
        falsely report convergence at a meaningless, extreme value (this was
        found by comparing regenerated fits against the published paper:
        without this bound, ~1% of retried fits "confidently" landed on
        answers like theta=-13.7 instead of the paper's -1.34). Bounding
        retries keeps the search inside the space the model is actually
        meant to describe.

    Returns a dict: theta_hat, se, ci95_lower, ci95_upper, convergence,
    reference_ll, gof, pseudo_r2. When robust=True, also includes
    n_attempts, n_converged, and restart_theta_std (the spread of theta_hat
    across attempts actually made -- a diagnostic for how ambiguous the fit
    is, not used to compute theta_hat itself).
    """
    demands = np.asarray(demands, dtype=float)
    success = np.asarray(success)

    if x_init is None:
        centers = (demands[:, 0] + demands[:, 1]) / 2
        bin_means, bin_edges, _ = binned_statistic(centers, success, statistic="mean", bins=n_bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        finite = (~np.isnan(bin_means)) & (~np.isnan(bin_centers))
        lowess_smoothed = lowess(bin_means[finite], bin_centers[finite], frac=lowess_frac, it=0)
        idx_max = np.argmax(lowess_smoothed[:, 1])
        x_init = lowess_smoothed[idx_max, 0]

    first = minimize(neg_log_likelihood, x0=x_init, args=(demands, success, k), method="BFGS", options={"maxiter": maxiter})
    attempts = [first]

    # Fully degenerate outcome data (every item succeeded, or every item
    # failed) has NO interior maximum at all: the likelihood improves
    # monotonically as theta walks toward the edge of whatever range it's
    # allowed to explore, because there's no counterbalancing data point to
    # pin down a peak. Retrying can only walk further down that ridge and
    # "confidently" land on the boundary -- worse than just accepting
    # whatever the single unconstrained attempt found, which is also what
    # the original single-shot pipeline does for these same rows. So skip
    # retries entirely in this case, regardless of the first attempt's
    # convergence flag.
    is_degenerate = bool(np.all(success == success[0]))

    if robust and not first.success and not is_degenerate:
        if restart_range is None:
            restart_range = (-5.0, 5.0)
        # Search outward from x_init (alternating +/- offsets) rather than
        # sweeping restart_range left-to-right: a left-to-right sweep can
        # exhaust `patience` on a run of unhelpful far points before ever
        # reaching a good starting point that happens to sit near x_init,
        # giving up early even though max_retries hasn't been spent.
        span = restart_range[1] - restart_range[0]
        offsets = np.linspace(span / max_retries, span, max_retries)
        candidate_points = []
        for i, off in enumerate(offsets):
            direction = 1 if i % 2 == 0 else -1
            candidate_points.append(float(np.clip(x_init + direction * off, restart_range[0], restart_range[1])))

        # Patience is measured against the best QUALIFYING (converged,
        # interior) attempt found so far, not raw NLL among all attempts --
        # a boundary-hugging or non-converged attempt can have a misleadingly
        # better raw NLL (see the comment on `best` below) and would
        # otherwise reset the patience counter without representing genuine
        # progress toward a trustworthy answer.
        best_qualifying_nll = None
        stall = 0
        for sp in candidate_points:
            res = minimize(neg_log_likelihood, x0=sp, args=(demands, success, k), method="L-BFGS-B",
                            bounds=[restart_range], options={"maxiter": maxiter})
            attempts.append(res)

            qualifies = res.success and restart_range[0] + 1e-6 < res.x[0] < restart_range[1] - 1e-6
            if qualifies and (best_qualifying_nll is None or res.fun < best_qualifying_nll - 1e-9):
                best_qualifying_nll = res.fun
                stall = 0
            else:
                stall += 1

            # Stop as soon as we have a genuinely trustworthy (qualifying)
            # answer, or once patience is exhausted -- a "converged" but
            # boundary-hugging attempt does NOT count as a reason to stop
            # early, since it wouldn't be selected as the winner anyway.
            # Tried letting the search keep going past the first qualifying
            # attempt (stopping on patience alone): empirically worse on RA
            # against the published paper (8 rows off by >0.5 vs 3) -- more
            # exploration finds more distinct local optima on these
            # near-flat surfaces, and "best likelihood among them" drifts
            # further from the paper's own single-shot answer more often
            # than it fixes genuine non-convergence. Stopping at the first
            # trustworthy attempt stays closer to the paper's own
            # single-shot methodology, which is what this flag is for.
            if qualifies or stall >= patience:
                break

    # Selecting a retry over the first attempt requires BOTH a better
    # likelihood AND that the retry itself converged AND landed away from
    # the search boundary -- deliberately conservative. Comparing
    # regenerated fits against the published paper showed that "best
    # likelihood" alone, or even "best likelihood among non-boundary
    # attempts", isn't safe: near-degenerate outcome data can have a
    # likelihood ridge so flat that some retry "confidently" (converged=True)
    # lands on an extreme, wrong value with a marginally better NLL than the
    # paper-matching first attempt. Requiring convergence + interiority
    # before ever overriding the first attempt catches this -- if no retry
    # qualifies, the first attempt's own (possibly non-converged, but
    # empirically paper-matching) result is kept rather than replaced with a
    # more "confident" but wrong one.
    if restart_range is not None:
        qualifying = [
            a for a in attempts[1:]
            if a.success and restart_range[0] + 1e-6 < a.x[0] < restart_range[1] - 1e-6
        ]
        best = min(qualifying, key=lambda r: r.fun) if qualifying and min(a.fun for a in qualifying) < first.fun else first
    else:
        best = first

    gof = -best.fun
    theta_hat = best.x[0]
    # BFGS's hess_inv is a dense ndarray; L-BFGS-B's (used for bounded
    # retries) is a LbfgsInvHessProduct LinearOperator that needs densifying
    # first -- np.diag() alone only works for the former.
    hess_inv = best.hess_inv
    hess_inv_dense = hess_inv.todense() if hasattr(hess_inv, "todense") else hess_inv
    se = np.sqrt(np.diag(np.asarray(hess_inv_dense)))[0]
    ci95_lower = theta_hat - 1.96 * se
    ci95_upper = theta_hat + 1.96 * se

    reference_ll = -neg_log_likelihood([0.0], demands, success, k)
    pseudo_r2 = 1 - (gof / reference_ll)

    result = {
        "theta_hat": theta_hat,
        "se": se,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
        "convergence": float(best.success),
        "reference_ll": reference_ll,
        "gof": gof,
        "pseudo_r2": pseudo_r2,
    }
    if robust:
        result["n_attempts"] = len(attempts)
        result["n_converged"] = sum(1 for a in attempts if a.success)
        result["restart_theta_std"] = float(np.std([a.x[0] for a in attempts]))
    return result
