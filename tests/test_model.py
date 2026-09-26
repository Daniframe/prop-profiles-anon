import warnings

import numpy as np
import pytest

from src.propensity.surfaces.model import two_sided_sigma


def _raw_exp_formula(x, x1, x2, k1, k2, min_width=0.1, rho=2.0):
    """Reproduces the pre-fix return statement of the original
    prod_sigma_normalised_final (raw exp, no expit), for regression comparison.
    Uses the corrected (outward) widening direction, matching model.py, so
    this isolates just the expit-vs-raw-exp difference."""
    max_exp = 700
    w0 = x2 - x1
    w = max(min_width, w0)
    x1 = x1 - (w - w0) / 2
    x2 = x2 + (w - w0) / 2

    exp_term = np.exp(np.clip(rho / w, -max_exp, max_exp))
    k1 = k1 + exp_term - 1
    k2 = k2 + exp_term - 1

    if np.isclose(k1, k2):
        exponent = np.clip(-k1 * (x2 - x1) / 2, -max_exp, max_exp)
        A = (1 + np.exp(exponent)) ** 2
    else:
        from src.propensity.surfaces.model import get_prod_normalized_A
        A = get_prod_normalized_A(x1, x2, k1, k2)

    return A / ((1 + np.exp(-k1 * (x - x1))) * (1 + np.exp(k2 * (x - x2))))


@pytest.mark.parametrize("k", [0.5, 1.0, 3.0])
@pytest.mark.parametrize("width", [0.01, 0.1, 0.5, 2.0, 10.0])
def test_peak_equals_one_at_midpoint(k, width):
    x1, x2 = -width / 2, width / 2
    midpoint = (x1 + x2) / 2
    p = two_sided_sigma(midpoint, x1, x2, k, k)
    assert p == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("x1,x2,k1,k2,x", [
    (-1.0, 1.0, 1.0, 1.0, 0.0),
    (-1.0, 1.0, 1.0, 1.0, 0.7),
    (-3.0, 2.0, 0.8, 1.2, -5.0),
    (0.0, 5.0, 2.0, 2.0, 4.0),
])
def test_bugfix_matches_original_on_normal_inputs(x1, x2, k1, k2, x):
    fixed = two_sided_sigma(x, x1, x2, k1, k2)
    original = _raw_exp_formula(x, x1, x2, k1, k2)
    assert fixed == pytest.approx(original, rel=1e-9, abs=1e-12)


def test_bugfix_avoids_overflow_warning_on_extreme_inputs():
    # A very narrow window drives k1/k2 up via e^(rho/w); evaluating well
    # outside the window makes the raw-exp formula's intermediate exp() calls
    # overflow to inf (raising a RuntimeWarning), even though it happens to
    # still divide out to the same (correct, saturated) value here. The
    # expit-based fix reaches that value without ever overflowing.
    x1, x2, k1, k2, x = -0.05, 0.05, 1.0, 1.0, -1.0

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        original = _raw_exp_formula(x, x1, x2, k1, k2)
    assert any(issubclass(w.category, RuntimeWarning) and "overflow" in str(w.message)
               for w in caught)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fixed = two_sided_sigma(x, x1, x2, k1, k2)
    assert not any(issubclass(w.category, RuntimeWarning) and "overflow" in str(w.message)
                   for w in caught)

    assert fixed == pytest.approx(original, abs=1e-9)
    assert 0.0 <= fixed <= 1.0


def test_min_width_floor_widens_narrow_interval():
    # Below min_width, the interval is symmetrically widened before
    # evaluation, so probability at the *original* (pre-floor) bounds is
    # pulled above 0.5 (they now sit inside the widened effective interval).
    x1, x2 = -0.01, 0.01
    p_at_original_bound = two_sided_sigma(x2, x1, x2, 1.0, 1.0, min_width=0.1)
    assert p_at_original_bound > 0.5


def test_symmetry_around_midpoint():
    x1, x2, k = -2.0, 3.0, 1.0
    m = (x1 + x2) / 2
    for d in (0.1, 0.5, 1.5):
        assert two_sided_sigma(m - d, x1, x2, k, k) == pytest.approx(
            two_sided_sigma(m + d, x1, x2, k, k), rel=1e-9
        )
