"""CLI entry point for deriving a propensity point estimate, curve, and
surface for one model, given demand annotations and outcomes.

Same interface as the original get_prop_point.py script (same flags, same
input file formats, same stdout line, same output filenames), now built on
top of src/propensity/*.
"""

import os
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.propensity.common import load_config
from src.propensity.surfaces import data, mle, curves, surfaces, plotting


def main():
    config = load_config("surfaces")
    prop_cfg = config.get("propensity", {})

    n_bins = prop_cfg.get("n_bins", 20)
    lowess_frac = prop_cfg.get("lowess_frac", 0.4)
    maxiter = prop_cfg.get("maxiter", 500)
    k_default = prop_cfg.get("k_default", 1.0)

    parser = argparse.ArgumentParser(
        description="Obtain propensity point of model given demands and outcomes"
    )
    parser.add_argument("--demands_filename", type=str)
    parser.add_argument("--outcomes_filename", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--r1", type=int, default=-3)
    parser.add_argument("--r2", type=int, default=3)
    parser.add_argument("--incited_prop", type=str, default=None)
    parser.add_argument("--plot_folder", type=str)
    args = parser.parse_args()

    df_demands = data.load_demands_single_dim(args.demands_filename)
    df_outcomes = data.load_outcomes(args.outcomes_filename)
    demands, success = data.build_arrays(df_demands, df_outcomes, args.model_name)

    fit = mle.fit_theta(demands, success, k=k_default, n_bins=n_bins,
                         lowess_frac=lowess_frac, maxiter=maxiter)

    true_x = float("nan") if args.incited_prop is None else float(args.incited_prop)
    fitted_x = fit["theta_hat"]
    ci95_lower = fit["ci95_lower"]
    ci95_upper = fit["ci95_upper"]

    print(true_x, fitted_x, ci95_lower, ci95_upper, fit["convergence"],
          fit["reference_ll"], fit["gof"], sep=",")

    os.makedirs(args.plot_folder, exist_ok=True)

    # PROP CURVE (uses a jittered re-binning for display clarity, matching
    # get_prop_point.py's behavior -- the fit itself uses the unjittered bins
    # internally, inside fit_theta, for its initial guess).
    curve = curves.build_empirical_curve(demands, success, n_bins=n_bins,
                                          lowess_frac=lowess_frac, jitter=True)
    plotting.plot_prop_curve(curve, fitted_x, ci95_lower, ci95_upper,
                              incited_prop=args.incited_prop, r1=args.r1, r2=args.r2)
    plt.savefig(f"{args.plot_folder}/PC_{args.model_name}.png", dpi=100)
    plt.close()

    # PROP HEATMAP
    surface = surfaces.build_empirical_surface(demands, success, args.r1, args.r2)
    plotting.plot_prop_surface(surface, fitted_x, ci95_lower, ci95_upper, args.model_name)
    plt.tight_layout()
    plt.savefig(f"{args.plot_folder}/HM_{args.model_name}.png", dpi=100)
    plt.close()


if __name__ == "__main__":
    main()
