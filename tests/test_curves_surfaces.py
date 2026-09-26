import numpy as np

from src.propensity.surfaces.curves import build_empirical_curve
from src.propensity.surfaces.surfaces import build_empirical_surface


def _simulate(n_items=200, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, n_items)
    half_widths = rng.uniform(0.5, 1.5, n_items)
    demands = np.column_stack([centers - half_widths, centers + half_widths])
    success = rng.binomial(1, 0.5, n_items)
    return demands, success


def test_empirical_curve_shapes_and_ranges():
    demands, success = _simulate()
    curve = build_empirical_curve(demands, success, n_bins=20)

    assert len(curve["bin_centers"]) == 20
    assert len(curve["bin_means"]) == 20

    finite_means = curve["bin_means"][~np.isnan(curve["bin_means"])]
    assert np.all((finite_means >= 0) & (finite_means <= 1))
    assert np.all((curve["lowess_y"] >= 0) & (curve["lowess_y"] <= 1))
    assert len(curve["lowess_x"]) == len(curve["lowess_y"])


def test_empirical_curve_jitter_changes_centers():
    demands, success = _simulate()
    curve_plain = build_empirical_curve(demands, success, n_bins=20, jitter=False)
    curve_jittered = build_empirical_curve(demands, success, n_bins=20, jitter=True, seed=42)
    assert not np.allclose(curve_plain["bin_centers"], curve_jittered["bin_centers"])


def test_empirical_surface_matches_manual_groupby():
    demands = np.array([
        [-1, 1],
        [-1, 1],
        [0, 2],
        [-2, 0],
    ], dtype=float)
    success = np.array([1, 0, 1, 0])

    surface = build_empirical_surface(demands, success, r1=-2, r2=2)

    # (b_u=1, b_l=-1): two observations, success=[1,0] -> count=2, prob=0.5
    assert surface["counts"].loc[1, -1] == 2
    assert surface["prob"].loc[1, -1] == 0.5

    # (b_u=2, b_l=0): one observation, success=1 -> count=1, prob=1.0
    assert surface["counts"].loc[2, 0] == 1
    assert surface["prob"].loc[2, 0] == 1.0

    # (b_u=0, b_l=-2): one observation, success=0 -> count=1, prob=0.0
    assert surface["counts"].loc[0, -2] == 1
    assert surface["prob"].loc[0, -2] == 0.0

    # a never-observed but structurally valid cell (b_l <= b_u) is 0 count, NaN prob
    assert surface["counts"].loc[2, -2] == 0
    assert np.isnan(surface["prob"].loc[2, -2])
