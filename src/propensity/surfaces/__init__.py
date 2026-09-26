"""Sector 4: derivation of propensity surfaces and curves (paper stage 3).

- model: canonical Eq. 2-5 propensity model (two_sided_sigma).
- variants: exploratory alternatives from early development (reference only).
- data: loading demand annotations (JSONL/CSV, single- or multi-dimension) and outcomes (wide CSV).
- mle: Eq. 6 maximum-likelihood fitting of theta.
- curves: empirical propensity curve construction.
- surfaces: empirical propensity surface construction.
- plotting: matplotlib figures for both.
- aggregate: batch-fit theta over multiple models/levels into a derived-propensity table.
"""

from .model import two_sided_sigma
from .mle import fit_theta, neg_log_likelihood
from .data import (
    load_demands_single_dim,
    load_demands_multi_dim,
    select_dim_demands,
    load_outcomes,
    build_arrays,
)
from .curves import build_empirical_curve
from .surfaces import build_empirical_surface
from .plotting import plot_prop_curve, plot_prop_surface
from .aggregate import derive_propensity_table

__all__ = [
    "two_sided_sigma",
    "fit_theta",
    "neg_log_likelihood",
    "load_demands_single_dim",
    "load_demands_multi_dim",
    "select_dim_demands",
    "load_outcomes",
    "build_arrays",
    "build_empirical_curve",
    "build_empirical_surface",
    "plot_prop_curve",
    "plot_prop_surface",
    "derive_propensity_table",
]
