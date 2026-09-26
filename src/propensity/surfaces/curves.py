"""Empirical propensity curve construction ("PROP CURVE" data prep): bins
observed success by interval center and LOWESS-smooths it. Ported from
get_prop_point.py.
"""

import numpy as np
from scipy.stats import binned_statistic
from statsmodels.nonparametric.smoothers_lowess import lowess

def build_empirical_curve(
    demands,
    success,
    n_bins = 20,
    lowess_frac = 0.4,
    jitter = False,
    seed = None
    ):
    
    """
    demands: (N, 2) array-like of (b_l, b_u) pairs.
    success: (N,) array-like of binary outcomes.
    jitter: if True, applies a +/-0.25 uniform jitter to demands before
        computing centers (matches get_prop_point.py's second, jittered pass
        used only to declutter the heatmap overlay; off by default since it's
        cosmetic and not part of the fit).

    Returns a dict: bin_centers, bin_means, lowess_x, lowess_y.
    """
    demands = np.asarray(demands, dtype=float)
    success = np.asarray(success)

    if jitter:
        rng = np.random.default_rng(seed)
        demands = demands + 0.5 * (rng.random(demands.shape) - 0.5)

    centers = (demands[:, 0] + demands[:, 1]) / 2

    bin_means, bin_edges, _ = binned_statistic(centers, success, statistic="mean", bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    finite = (~np.isnan(bin_means)) & (~np.isnan(bin_centers))
    lowess_smoothed = lowess(bin_means[finite], bin_centers[finite], frac=lowess_frac, it=0)

    return {
        "bin_centers": bin_centers,
        "bin_means": bin_means,
        "lowess_x": lowess_smoothed[:, 0],
        "lowess_y": lowess_smoothed[:, 1],
    }
