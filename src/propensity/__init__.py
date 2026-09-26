"""propensity: reproduction of the full pipeline behind the accompanying paper.

Subpackages, one per pipeline sector:
- common: cross-sector utilities (config loading, reproducible IDs, I/O, LLM clients).
- generation: Sector 1, synthetic/external benchmark construction.
- annotation: Sector 2, rubric-based demand-interval annotation.
- inference: Sector 3, instance-level model outcomes across incitement levels.
- surfaces: Sector 4, propensity curve/surface fitting (Eq. 2-6).
- predictability: Sector 5, capability+propensity predictive-power assessment.

Re-exports the public API of each implemented sector at the top level.
"""

from .surfaces import (
    two_sided_sigma,
    fit_theta,
    neg_log_likelihood,
    load_demands_single_dim,
    load_demands_multi_dim,
    select_dim_demands,
    load_outcomes,
    build_arrays,
    build_empirical_curve,
    build_empirical_surface,
    plot_prop_curve,
    plot_prop_surface,
    derive_propensity_table,
)

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
