"""Batch theta-fitting across models x incitement levels, producing the same
table shape as the original Derived_propensities_{dataset}.csv. Factors
get_all_prop_points.py's subprocess-per-(model,level) loop into an in-process
function calling mle.fit_theta directly.
"""

from typing import List, Optional, Union

import pandas as pd

from .data import build_arrays
from .mle import fit_theta

DEFAULT_LEVELS = ["-3", "-2", "-1", "0", "+1", "+2", "+3", None]


def derive_propensity_table(
    demands_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    models: List[str],
    code: str,
    levels: List[Optional[Union[str, int]]] = DEFAULT_LEVELS,
    k: float = 1.0,
    n_bins: int = 20,
    lowess_frac: float = 0.4,
    maxiter: int = 500,
    robust: bool = False,
    max_retries: int = 20,
    patience: int = 2,
) -> pd.DataFrame:
    """Fits theta for every (model, level) pair, skipping any pair whose
    outcome column isn't present in outcomes_df (e.g. a model that wasn't
    run at a given level) or that has zero rows after the demands/outcomes
    join.

    levels: incitement levels to test; None means the "unprompted" column
        {model}_{code}_outcome (no level suffix) -- matches
        get_all_prop_points.py's LEVELS list, which included a bare "" entry.
    robust, max_retries, patience: forwarded to mle.fit_theta -- robust=False
        (default) is a single BFGS run per pair, unchanged from before this
        parameter existed. robust=True only spends extra attempts on pairs
        whose first attempt doesn't converge, escalating until convergence,
        no further improvement for `patience` attempts, or `max_retries` is
        hit (see fit_theta's docstring for the full policy and why the
        winner is picked by best likelihood rather than averaging).

    Returns a DataFrame with columns Model, Incited propensity, Obtained
    propensity, Lower 95CI, Upper 95CI, Converges, ReferenceLL, EstimationLL
    -- the same schema as Derived_propensities_{dataset}.csv. When
    robust=True, also includes NAttempts/NConverged/RestartThetaStd.
    """
    rows = []
    for model in models:
        for level in levels:
            model_name = f"{model}_{code}" if level is None else f"{model}_{code}_{level}"
            if f"{model_name}_outcome" not in outcomes_df.columns:
                continue

            demands, success = build_arrays(demands_df, outcomes_df, model_name)
            if len(demands) == 0:
                continue

            fit = fit_theta(demands, success, k=k, n_bins=n_bins, lowess_frac=lowess_frac,
                             maxiter=maxiter, robust=robust, max_retries=max_retries, patience=patience)
            incited_prop = None if level is None else float(level)

            row = {
                "Model": model,
                "Incited propensity": incited_prop,
                "Obtained propensity": fit["theta_hat"],
                "Lower 95CI": fit["ci95_lower"],
                "Upper 95CI": fit["ci95_upper"],
                "Converges": fit["convergence"],
                "ReferenceLL": fit["reference_ll"],
                "EstimationLL": fit["gof"],
            }
            if robust:
                row["NAttempts"] = fit["n_attempts"]
                row["NConverged"] = fit["n_converged"]
                row["RestartThetaStd"] = fit["restart_theta_std"]
            rows.append(row)

    columns = ["Model", "Incited propensity", "Obtained propensity", "Lower 95CI",
               "Upper 95CI", "Converges", "ReferenceLL", "EstimationLL"]
    if robust:
        columns += ["NAttempts", "NConverged", "RestartThetaStd"]
    return pd.DataFrame(rows, columns=columns)
