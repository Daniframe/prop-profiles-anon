"""Exploratory bell-curve formulations from early development of the Eq. 5
propensity model (see src/propensity/model.py for the canonical one actually
used by the pipeline). Ported verbatim from the original scrappy `curves.py`
for reference / possible appendix reproduction -- not used by mle.py, curves.py,
or surfaces.py, and no bugfixes are applied here.
"""

import numpy as np
from scipy.optimize import fmin, minimize

from .model import (
    get_prod_normalized_A_if_ks_equal,
    get_prod_normalized_A,
    get_peak_value,
)

def std_sigma(x, x0, k, L=1):
    """Standard logistic function."""
    return L / (1 + np.exp(-k * (x - x0)))

def falling_logistic(x, x0, k, L=1):
    """High at low x, low at high x (k > 0, drops as x increases past x0)."""
    return L / (1 + np.exp(k * (x - x0)))

def diff_sigma(x, x1, x2, k1, k2, L1=1, L2=1):
    """Difference of two logistic functions: bell-shaped function."""
    return std_sigma(x, x1, k1, L1) - std_sigma(x, x2, k2, L2)

def prod_sigma(x, x1, x2, k1, k2, A=1):
    """Product of two logistic functions: bell-shaped function."""
    return A / ((1 + np.exp(-k1 * (x - x1))) * (1 + np.exp(k2 * (x - x2))))

def prod_sigma_normalised(x, x1, x2, k1, k2):
    """Product of two logistic functions, normalised so the peak equals 1."""
    if k1 == k2:
        A = get_prod_normalized_A_if_ks_equal(x1, x2, k1, k2)
    else:
        A = get_prod_normalized_A(x1, x2, k1, k2)
    return A / ((1 + np.exp(-k1 * (x - x1))) * (1 + np.exp(k2 * (x - x2))))

def get_diff_normalized_A_if_ks_equal(x1, x2, k1, k2):
    x_peak = (x1 + x2) / 2
    peak_height = std_sigma(x_peak, x1, k1) - std_sigma(x_peak, x2, k2)
    return 1 / peak_height

def get_diff_normalized_A(x1, x2, k1, k2):
    func = lambda x: std_sigma(x, x1, k1) - std_sigma(x, x2, k2)
    x_max = fmin(func, (x1 + x2) / 2, disp=False)[0]
    max_val_with_A1 = -func(x_max)
    return 1.0 / max_val_with_A1

def diff_sigma_normalised(x, x1, x2, k1, k2):
    """Difference of two logistic functions, normalised so the peak equals 1."""
    if k1 == k2:
        A = get_diff_normalized_A_if_ks_equal(x1, x2, k1, k2)
    else:
        A = get_diff_normalized_A(x1, x2, k1, k2)
    return std_sigma(x, x1, k1, A) - std_sigma(x, x2, k2, A)

def prod_sigma_normalised_5(x, x1, x2, k1, k2):
    """Like prod_sigma_normalised, but slopes are bumped ad-hoc to pull the
    interval bounds closer to 0.5."""
    k1 = max(k1, 5 / (x2 - x1 + 0.0001))
    k2 = max(k2, 5 / (x2 - x1 + 0.0001))

    if k1 == k2:
        A = get_prod_normalized_A_if_ks_equal(x1, x2, k1, k2)
    else:
        A = get_prod_normalized_A(x1, x2, k1, k2)
    return A / ((1 + np.exp(-k1 * (x - x1))) * (1 + np.exp(k2 * (x - x2))))

def prod_sigma_normalised_L(x, x1, x2, k1, k2):
    if k1 == k2:
        A = (1 + np.exp(-k1 * (x2 - x1) / 2)) ** 2  # Lorenzo's equation
    else:
        A = get_prod_normalized_A(x1, x2, k1, k2)
    return A / ((1 + np.exp(-k1 * (x - x1))) * (1 + np.exp(k2 * (x - x2))))

def prod_sigma_normalised_old(x, x1, x2, k1, k2):
    """Earlier version of the final model (see model.two_sided_sigma), kept for
    reference; unlike the final version this recomputes A/k1/k2 without
    clipping the e^(rho/w) exponent, and prints debug info."""
    rho = 2

    w0 = x2 - x1
    min_w = 0.1  # with this small a size, k1 can be very large and A infinite
    w = max(min_w, w0)
    x1 = x1 + (w - w0) / 2
    x2 = x2 - (w - w0) / 2

    num = rho
    k1 = k1 + np.exp(num / w) - 1
    k2 = k2 + np.exp(num / w) - 1

    if k1 == k2:
        A = (1 + np.exp(-k1 * (x2 - x1) / 2)) ** 2  # Lorenzo's equation
    else:
        A = get_prod_normalized_A(x1, x2, k1, k2)

    print(f"A: {A}, x1: {x1}, x2: {x2}, k1: {k1}  ")
    return A / ((1 + np.exp(-k1 * (x - x1))) * (1 + np.exp(k2 * (x - x2))))

def centred_gaussian(x, x1, x2):
    xc = (x1 + x2) / 2
    w = x2 - x1
    l2 = np.log(2)
    return np.exp(-4 * l2 * ((x - xc) / w) ** 2)

def logistic_bump(x, x1, x2, n=1):
    xc = (x1 + x2) / 2
    w = x2 - x1
    return 1.0 / (1.0 + (2 * (x - xc) / w) ** (2 * n))

def original_function(x, x1, x2, k1, k2, A):
    return A / ((1 + np.exp(-k1 * (x - x1))) * (1 + np.exp(k2 * (x - x2))))

def find_parameters(x1_prime, x2_prime, initial_guess_params):
    """Finds (x1, x2, k1, k2) such that f(x1')=0.5, f(x2')=0.5, and max=1."""

    def objective_function(params):
        x1_val, x2_val, k1_val, k2_val = params
        if x1_val >= x2_val:
            return 1e10

        peak_val_A1 = get_peak_value(x1_val, x2_val, k1_val, k2_val)
        A_val = 1.0 / peak_val_A1

        f_x1_prime = original_function(x1_prime, x1_val, x2_val, k1_val, k2_val, A_val)
        f_x2_prime = original_function(x2_prime, x1_val, x2_val, k1_val, k2_val, A_val)

        error = (f_x1_prime - 0.5) ** 2 + (f_x2_prime - 0.5) ** 2
        return error

    result = minimize(
        objective_function,
        initial_guess_params,
        method="L-BFGS-B",
        bounds=[(None, None), (None, None), (0.1, None), (0.1, None)],
    )

    if result.success:
        x1_opt, x2_opt, k1_opt, k2_opt = result.x
        A_opt = 1.0 / get_peak_value(x1_opt, x2_opt, k1_opt, k2_opt)
        return x1_opt, x2_opt, k1_opt, k2_opt, A_opt, result.fun
    else:
        raise ValueError(f"Optimization failed: {result.message}")

def prod_sigma_normalised_ok(x, x1, x2, k1, k2):
    """Product of two logistics, normalised so the peak is 1 AND the interval
    bounds are exactly at 0.5 (found by numerical optimisation, see
    find_parameters)."""
    w0 = x2 - x1
    min_w = 0.01  # a minimum width is needed, otherwise it may not converge
    w = max(min_w, w0)
    x1 = x1 + (w - w0) / 2
    x2 = x2 - (w - w0) / 2

    x1_prime_target = x1
    x2_prime_target = x2
    initial_guess = [
        (x1_prime_target + x2_prime_target) / 2 - 0.5,
        (x1_prime_target + x2_prime_target) / 2 + 0.5,
        k1,
        k2,
    ]

    x1_final, x2_final, k1_final, k2_final, A_final, final_error = find_parameters(
        x1_prime_target, x2_prime_target, initial_guess
    )

    return original_function(x, x1_final, x2_final, k1_final, k2_final, A_final)
