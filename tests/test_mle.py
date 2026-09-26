import numpy as np
import pytest

from src.propensity.surfaces.model import two_sided_sigma
from src.propensity.surfaces.mle import fit_theta, neg_log_likelihood


def _simulate(theta_true, n_items=250, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, n_items)
    half_widths = rng.uniform(0.5, 2.5, n_items)
    x1 = centers - half_widths
    x2 = centers + half_widths
    demands = np.column_stack([x1, x2])

    probs = np.array([two_sided_sigma(theta_true, a, b, 1.0, 1.0) for a, b in demands])
    success = rng.binomial(1, probs)
    return demands, success


def test_fit_theta_recovers_known_theta():
    theta_true = -1.5
    demands, success = _simulate(theta_true, seed=1)
    fit = fit_theta(demands, success)
    assert abs(fit["theta_hat"] - theta_true) < 0.4


def test_ci_bounds_ordering():
    demands, success = _simulate(0.5, seed=2)
    fit = fit_theta(demands, success)
    assert fit["ci95_lower"] < fit["theta_hat"] < fit["ci95_upper"]


def test_neg_log_likelihood_matches_hand_computation():
    demands = np.array([[-1.0, 1.0], [0.0, 2.0]])
    success = np.array([1, 0])

    p0 = two_sided_sigma(0.3, -1.0, 1.0, 1.0, 1.0)
    p1 = two_sided_sigma(0.3, 0.0, 2.0, 1.0, 1.0)
    expected = -np.log(p0 * (1 - p1))

    assert neg_log_likelihood(0.3, demands, success) == pytest.approx(expected, rel=1e-9)


def _simulate_near_ceiling(n_items=300, seed=0, n_failures=15):
    """Near-degenerate data: almost every item succeeds regardless of demand
    interval, with only a handful of scattered failures -- the same failure
    mode observed in the real Ul benchmark's unstable (model, level) fits,
    where the likelihood surface is nearly flat and a single BFGS run can
    land in an arbitrary local optimum."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, n_items)
    half_widths = rng.uniform(0.5, 1.5, n_items)
    demands = np.column_stack([centers - half_widths, centers + half_widths])
    success = np.ones(n_items, dtype=int)
    fail_idx = rng.choice(n_items, size=n_failures, replace=False)
    success[fail_idx] = 0
    return demands, success


def test_fit_theta_default_robust_false_returns_same_keys_as_before():
    demands, success = _simulate(0.0, seed=3)
    fit = fit_theta(demands, success)
    assert set(fit.keys()) == {
        "theta_hat", "se", "ci95_lower", "ci95_upper",
        "convergence", "reference_ll", "gof", "pseudo_r2",
    }


def test_fit_theta_robust_true_adds_diagnostic_keys():
    demands, success = _simulate(0.0, seed=3)
    fit = fit_theta(demands, success, robust=True)
    assert {"n_attempts", "n_converged", "restart_theta_std"} <= set(fit.keys())


def test_fit_theta_robust_skips_retries_when_first_attempt_converges():
    """A well-conditioned fit converges on the first (LOWESS-seeded) attempt,
    so robust=True should spend no extra attempts on it."""
    demands, success = _simulate(0.5, seed=2)
    single = fit_theta(demands, success)
    assert single["convergence"] == 1.0  # sanity check this fixture is well-behaved

    robust_fit = fit_theta(demands, success, robust=True, max_retries=20, patience=2)
    assert robust_fit["n_attempts"] == 1
    assert robust_fit["theta_hat"] == single["theta_hat"]


def test_fit_theta_robust_only_retries_when_first_attempt_fails():
    demands, success = _simulate_near_ceiling(seed=7)
    single = fit_theta(demands, success)
    robust_fit = fit_theta(demands, success, robust=True, max_retries=20, patience=2)

    if single["convergence"] == 1.0:
        assert robust_fit["n_attempts"] == 1
    else:
        assert robust_fit["n_attempts"] > 1


def test_fit_theta_robust_is_never_worse_than_single_start():
    """The single-start x_init is always the first attempt, so the
    best-likelihood selection can only match or beat (lower gof/nll, i.e.
    higher or equal likelihood) the single-start result -- never worse."""
    demands, success = _simulate_near_ceiling(seed=7)
    single = fit_theta(demands, success)
    robust_fit = fit_theta(demands, success, robust=True, max_retries=20, patience=2)
    assert robust_fit["gof"] >= single["gof"] - 1e-9


def test_fit_theta_robust_respects_max_retries_cap():
    demands, success = _simulate_near_ceiling(seed=7)
    robust_fit = fit_theta(demands, success, robust=True, max_retries=5, patience=100)
    assert robust_fit["n_attempts"] <= 1 + 5


def test_fit_theta_robust_reports_convergence_spread_diagnostic():
    demands, success = _simulate_near_ceiling(seed=11)
    robust_fit = fit_theta(demands, success, robust=True, max_retries=20, patience=2)
    assert robust_fit["n_converged"] <= robust_fit["n_attempts"]
    assert robust_fit["restart_theta_std"] >= 0.0


def _simulate_all_failure(n_items=200, seed=0):
    """Fully degenerate data: every single response fails, regardless of
    demand interval -- the real failure mode found in production (e.g. a
    model whose baseline/unprompted run scored 0/187). The likelihood has no
    interior maximum here: it improves monotonically as theta -> +-infinity
    (pushing the sigmoid to 0 everywhere), so an unconstrained retry can walk
    arbitrarily far down that tail and have BFGS falsely report convergence
    there. This is a regression test for exactly that bug."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, n_items)
    half_widths = rng.uniform(0.5, 1.5, n_items)
    demands = np.column_stack([centers - half_widths, centers + half_widths])
    success = np.zeros(n_items, dtype=int)
    return demands, success


def test_fit_theta_robust_skips_retries_entirely_on_degenerate_data():
    """Regression test (two-stage bug): the first fix (bounding retries to
    restart_range) only capped the divergence at the boundary (e.g. exactly
    -5.0) instead of preventing it -- because for fully degenerate data
    *any* point further from the data is "better" in raw likelihood, all the
    way to whatever boundary exists. Comparing regenerated fits against the
    published paper caught this: rows that should read ~-1.34 (matching both
    the original pipeline and the paper) were instead landing exactly on the
    retry boundary. The real fix is to not retry AT ALL on degenerate data --
    robust=True must return exactly the single unconstrained attempt,
    matching the non-robust/original pipeline behavior for these rows."""
    demands, success_all_fail = _simulate_all_failure(seed=1)
    success_all_succeed = 1 - success_all_fail

    for success in (success_all_fail, success_all_succeed):
        single = fit_theta(demands, success)
        robust_fit = fit_theta(demands, success, robust=True, max_retries=20, patience=2)
        assert robust_fit["n_attempts"] == 1
        assert robust_fit["theta_hat"] == single["theta_hat"]


def test_fit_theta_robust_respects_custom_restart_range_bounds():
    """On non-degenerate near-ceiling data (where retries legitimately run),
    a custom restart_range still bounds where retries can land."""
    demands, success = _simulate_near_ceiling(seed=7)
    robust_fit = fit_theta(demands, success, robust=True, max_retries=20, patience=2, restart_range=(-3.0, 3.0))
    assert -3.0 <= robust_fit["theta_hat"] <= 3.0
