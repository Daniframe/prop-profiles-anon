"""Canonical propensity model (Eq. 2-5 of the paper).

`two_sided_sigma` is the normalized two-sided 2x2PL model used throughout the
pipeline (paper Eq. 5). It is a direct port of `prod_sigma_normalised_final`
from the original scrappy `curves.py`, with one numerical fix: the original
computed numerically-stable sigmoid terms via `expit` but then discarded them,
returning a raw `exp`-based product that can overflow for narrow windows /
large slopes. This version returns the algebraically identical, numerically
stable `A * sigma1 * sigma2` instead.

See src/propensity/variants.py for the other exploratory formulations that
were tried before landing on this one.
"""

import numpy as np
from scipy.special import expit
from scipy.optimize import fmin, minimize_scalar

MAX_EXP = 700  # np.exp(709) is near float64 overflow

def get_prod_normalized_A_if_ks_equal(x1, x2, k1, k2):
    """Closed-form normalization A so the peak (at k1==k2) equals 1."""
    x_peak = (x1 + x2) / 2
    peak_height = 1 / ((1 + np.exp(-k1 * (x_peak - x1))) * (1 + np.exp(k2 * (x_peak - x2))))
    return 1 / peak_height

def get_prod_normalized_A(x1, x2, k1, k2):
    """Numerically finds A so the peak equals 1 when k1 != k2 (asymmetric slopes)."""
    func = lambda x: -1 / ((1 + np.exp(-k1 * (x - x1))) * (1 + np.exp(k2 * (x - x2))))
    x_max = fmin(func, (x1 + x2) / 2, disp=False)[0]
    max_val_with_A1 = -func(x_max)
    return 1.0 / max_val_with_A1

def get_peak_value(x1, x2, k1, k2):
    """Numerically finds the maximum value of the unnormalized (A=1) product-sigmoid."""
    func_A1 = lambda x: -1 / ((1 + np.exp(-k1 * (x - x1))) * (1 + np.exp(k2 * (x - x2))))
    result = minimize_scalar(func_A1, bounds=(x1, x2), method="bounded")
    return -result.fun

def two_sided_sigma(x, x1, x2, k1, k2, min_width=0.1, rho=2.0):
    """
    Eq. 5: normalized two-sided 2x2PL propensity model.

    x: propensity value(s) theta at which to evaluate P(success).
    x1, x2: demand interval bounds (b_l, b_u); x1 < x2.
    k1, k2: discrimination slopes (a_l, a_u); typically k1 == k2 == a.
    min_width: floor on (x2 - x1) before computing discrimination -- narrow
        annotated intervals are symmetrically widened outward to this floor
        to avoid the e^(rho/w) adjusted-discrimination term overflowing.
    rho: controls how much discrimination grows as the interval narrows
        (Eq. 3's numerator; adjusted slope = k + e^(rho/w) - 1).

    Returns P(success | x, x1, x2, k1, k2) in [0, 1]; peak of 1.0 at the
    interval midpoint by construction.
    """
    w0 = x2 - x1
    w = max(min_width, w0)
    x1 = x1 - (w - w0) / 2
    x2 = x2 + (w - w0) / 2

    exp_term = np.exp(np.clip(rho / w, -MAX_EXP, MAX_EXP))
    k1 = k1 + exp_term - 1
    k2 = k2 + exp_term - 1

    if np.isclose(k1, k2):
        exponent = np.clip(-k1 * (x2 - x1) / 2, -MAX_EXP, MAX_EXP)
        A = (1 + np.exp(exponent)) ** 2
    else:
        A = get_prod_normalized_A(x1, x2, k1, k2)

    if not np.isfinite(A):
        A = np.finfo(float).max / 10

    sigma1 = expit(k1 * (x - x1))
    sigma2 = expit(-k2 * (x - x2))

    return A * sigma1 * sigma2
