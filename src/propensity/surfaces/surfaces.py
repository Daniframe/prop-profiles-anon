"""Empirical propensity surface construction ("PROP HEATMAP" data prep):
aggregates observed success onto an integer (b_l, b_u) grid. Ported from
get_prop_point.py.
"""

import numpy as np
import pandas as pd


def build_empirical_surface(demands, success, r1, r2):
    """
    demands: (N, 2) array-like of (b_l, b_u) pairs (rounded to int grid cells).
    success: (N,) array-like of binary outcomes.
    r1, r2: inclusive integer range of the (b_l, b_u) grid.

    Returns a dict:
      counts: DataFrame indexed by b_u, columned by b_l -- observation count per cell.
      prob: DataFrame (same shape) -- mean success rate per cell (NaN where no data).
      prob_plot: prob with cells that are valid (b_l <= b_u) but unobserved filled
        with the sentinel -0.1, for heatmap rendering via cmap.set_under.
      mask: boolean ndarray marking the cells filled with the sentinel.
      grid_vals: the integer grid values used for both axes.
    """
    demands = np.asarray(demands, dtype=float)
    success = np.asarray(success)

    bl = demands[:, 0]
    bu = demands[:, 1]

    df = pd.DataFrame({
        "b_l": bl.astype(int),
        "b_u": bu.astype(int),
        "success": success,
    })

    grid_vals = np.arange(r1, r2 + 1)
    full_index = pd.MultiIndex.from_product([grid_vals, grid_vals])

    counts = (
        df.groupby(["b_u", "b_l"])
        .size()
        .reindex(full_index, fill_value=0)
        .unstack()
    )

    prob = (
        df.groupby(["b_u", "b_l"])["success"]
        .mean()
        .reindex(full_index)
        .unstack()
    )

    prob_plot = prob.copy().astype(float)
    invalid_triangle = counts.columns.values[None, :] <= counts.index.values[:, None]
    nans = prob_plot.isna().values
    mask = invalid_triangle & nans
    prob_plot[mask] = -0.1

    return {
        "counts": counts,
        "prob": prob,
        "prob_plot": prob_plot,
        "mask": mask,
        "grid_vals": grid_vals,
    }
