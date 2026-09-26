"""Matplotlib figures for propensity curves/surfaces ("PROP CURVE" /
"PROP HEATMAP" from get_prop_point.py). Consumes only the plain dict outputs
of curves.py/surfaces.py/mle.py -- never calls model.py or reads config
directly -- so the rest of the package stays testable without a display
backend. Neither function calls plt.show()/savefig(); the caller decides.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap


def plot_prop_curve(curve, theta_hat, ci95_lower, ci95_upper, incited_prop=None,
                     r1=None, r2=None, ax=None):
    """PROP CURVE: binned empirical success rate + LOWESS smooth, with vertical
    reference lines for the incited propensity (if given), the estimated
    theta_hat, and its 95% CI bounds.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.plot(curve["bin_centers"], curve["bin_means"], "ok", label="Binned fraction correct")
    ax.plot(curve["lowess_x"], curve["lowess_y"], "b-", label="LOWESS smooth (non-parametric)")

    if incited_prop is not None:
        ax.axvline(x=float(incited_prop), color="grey", linestyle="-", linewidth=2,
                   label=f"Incited propensity = {float(incited_prop):.2f}")

    ax.axvline(x=theta_hat, color="orange", linestyle="-", linewidth=2,
               label=f"Estimated propensity = {theta_hat:.2f}")
    ax.axvline(x=ci95_lower, color="orange", ls="--", lw=1,
               label="Estimated propensity\n" + rf"range: $[{ci95_lower:.2f}, {ci95_upper:.2f}]$")
    ax.axvline(x=ci95_upper, color="orange", ls="--", lw=1)

    ax.set_xlabel(r"Interval centre $(b_l+b_u)/2$")
    ax.set_ylabel("Fraction Success")
    ax.set_ylim(-0.05, 1.05)
    if r1 is not None and r2 is not None:
        ax.set_xlim(min(r1, ci95_lower) - 0.1, max(r2, ci95_upper) + 0.1)
    ax.legend()
    ax.grid(True)
    ax.set_axisbelow(True)
    return ax


def _get_line_vals(grid_vals, prop):
    """Coordinates (in heatmap cell-index space) of the b_l + b_u = 2*prop
    diagonal, clipped to the visible grid."""
    grid_min = grid_vals.min()
    n = len(grid_vals)

    x_vals = np.linspace(grid_min - 0.5, grid_vals.max() + 0.5, 10)
    y_vals = 2 * prop - x_vals

    x_plot = x_vals - grid_min + 0.5
    y_plot = y_vals - grid_min + 0.5

    mask = (x_plot >= 0) & (x_plot <= n) & (y_plot >= 0) & (y_plot <= n)
    return x_plot, y_plot, mask


def plot_prop_surface(surface, theta_hat, ci95_lower, ci95_upper, model_name, ax=None):
    """PROP HEATMAP: empirical success-rate heatmap over the (b_l, b_u) grid,
    annotated with observation counts, plus diagonal lines marking where the
    interval centre equals theta_hat (solid) and its CI95 bounds (dashed).
    """
    grid_vals = surface["grid_vals"]

    cmap = LinearSegmentedColormap.from_list(
        "red_yellow_green_no_white", ["#d7191c", "#FFFF99", "#1a9641"]
    )
    cmap.set_under(color="whitesmoke")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ax = sns.heatmap(
        surface["prob_plot"],
        cmap=cmap,
        vmin=0, vmax=1,
        square=True,
        annot=surface["counts"],
        fmt=".0f",
        cbar_kws={"label": "Rel. freq. of success", "fraction": 0.046, "pad": 0.04},
        ax=ax,
    )

    cbar = ax.collections[0].colorbar
    cbar.set_label("P(success)", fontsize=14) # pyright: ignore[reportOptionalMemberAccess]
    cbar.ax.tick_params(labelsize=12) # pyright: ignore[reportOptionalMemberAccess]

    ax.invert_yaxis()
    ax.set_xlabel(r"Lower limit ($b_l$)", fontsize=14)
    ax.set_ylabel(r"Upper limit ($b_u$)", fontsize=14)

    x_plot, y_plot, mask = _get_line_vals(grid_vals, theta_hat)
    ax.plot(x_plot[mask], y_plot[mask], color="black", lw=2,
            label=rf"Propensity level = {theta_hat:.2f}" + "\n" + rf"$[{ci95_lower:.2f}, {ci95_upper:.2f}]$")

    x_lower, y_lower, mask_lower = _get_line_vals(grid_vals, ci95_lower)
    x_upper, y_upper, mask_upper = _get_line_vals(grid_vals, ci95_upper)
    ax.plot(x_lower[mask_lower], y_lower[mask_lower], color="black", lw=1, ls="--")
    ax.plot(x_upper[mask_upper], y_upper[mask_upper], color="black", lw=1, ls="--")

    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.set_title(model_name)
    ax.legend(loc="best")
    return ax
