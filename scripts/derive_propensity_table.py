"""CLI entry point for Sector 4's batch driver: fits theta for every
model x incitement level, producing the same table shape as
the original Derived_propensities_{dataset}.csv -- replaces
get_all_prop_points.py's subprocess-per-(model,level) loop with one
in-process pass over src/propensity/surfaces/aggregate.py.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.propensity.common import load_config
from src.propensity.surfaces import data, aggregate


def main():
    config = load_config("surfaces")
    prop_cfg = config.get("propensity", {})

    parser = argparse.ArgumentParser(
        description="Fit theta for every model x incitement level and write a Derived_propensities table"
    )
    parser.add_argument("--demands_filename", type=str, required=True)
    parser.add_argument("--outcomes_filename", type=str, required=True)
    parser.add_argument("--code", type=str, required=True, help="Dimension code embedded in outcome column names, e.g. RA")
    parser.add_argument("--models", type=str, required=True, help="Comma-separated model shortcuts, e.g. 4o,llama33,gemma3")
    parser.add_argument("--levels", type=str, default="-3,-2,-1,0,+1,+2,+3,", help="Comma-separated levels; an empty entry means the unprompted/bare column")
    parser.add_argument("--robust", action="store_true",
                         help="Retry with extra starting points ONLY when the first BFGS attempt fails to converge, "
                              "stopping early once converged or once the fit stops improving (see mle.fit_theta)")
    parser.add_argument("--max-retries", type=int, default=20, help="--robust only: cap on additional attempts")
    parser.add_argument("--patience", type=int, default=2, help="--robust only: give up after this many non-improving additional attempts")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    demands_df = data.load_demands_single_dim(args.demands_filename)
    outcomes_df = data.load_outcomes(args.outcomes_filename)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    levels = [lvl if lvl != "" else None for lvl in args.levels.split(",")]

    table = aggregate.derive_propensity_table(
        demands_df, outcomes_df, models, args.code, levels=levels,
        k=prop_cfg.get("k_default", 1.0),
        n_bins=prop_cfg.get("n_bins", 20),
        lowess_frac=prop_cfg.get("lowess_frac", 0.4),
        maxiter=prop_cfg.get("maxiter", 500),
        robust=args.robust,
        max_retries=args.max_retries,
        patience=args.patience,
    )
    table.to_csv(args.out, index=False)
    print(f"Fit {len(table)} (model, level) pairs, wrote {args.out}")


if __name__ == "__main__":
    main()
