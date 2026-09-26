"""I/O matching the annotation/inference stages' actual output formats:
demand annotations as JSONL or CSV, model outcomes as a wide-format CSV (one
`{model_name}_outcome` column per model). Ported from get_prop_point.py.
"""

from pathlib import Path
from typing import List, Union

import pandas as pd


def _read_table(path: Union[str, Path]) -> pd.DataFrame:
    """Reads a JSONL or CSV file, format inferred from its extension."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension '{path.suffix}' (expected .jsonl or .csv)")


def load_demands_single_dim(
    path: Union[str, Path],
    lower_colname: str = "propensity_lower",
    upper_colname: str = "propensity_upper",
) -> pd.DataFrame:
    """Reads demand annotations for a single propensity dimension from a
    JSONL or CSV file; renames lower_colname/upper_colname to lower/upper."""
    return _read_table(path).rename(
        columns={lower_colname: "lower", upper_colname: "upper"}
    )


def load_demands_multi_dim(
    path: Union[str, Path],
    dim_names: List[str],
    lower_suffix: str = "_l",
    upper_suffix: str = "_u",
    question_id_col: str = "question_id",
) -> pd.DataFrame:
    """
    Reads demand annotations for several propensity dimensions at once from a
    single JSONL or CSV file, where each dimension `d` in dim_names has
    columns f"{d}{lower_suffix}" / f"{d}{upper_suffix}" -- e.g. for
    dim_names=["RA", "Ex", "Ul"] with the default suffixes, expects columns
    "RA_l"/"RA_u", "Ex_l"/"Ex_u", "Ul_l"/"Ul_u".

    Returns a single DataFrame with question_id_col plus two columns per
    dimension, renamed to the canonical f"{dim}_lower"/f"{dim}_upper" -- e.g.
    question_id, RA_lower, RA_upper, Ex_lower, Ex_upper, Ul_lower, Ul_upper
    -- regardless of what lower_suffix/upper_suffix the input file used.
    Use select_dim_demands() to pull out one dimension's (lower, upper)
    columns in the shape build_arrays/fit_theta expect.
    """
    df = _read_table(path)

    selected_cols = [question_id_col]
    rename_map = {}
    for dim in dim_names:
        lower_col = f"{dim}{lower_suffix}"
        upper_col = f"{dim}{upper_suffix}"
        missing = [c for c in (lower_col, upper_col) if c not in df.columns]
        if missing:
            raise ValueError(f"Dimension '{dim}': missing column(s) {missing}")
        selected_cols += [lower_col, upper_col]
        rename_map[lower_col] = f"{dim}_lower"
        rename_map[upper_col] = f"{dim}_upper"

    return df[selected_cols].rename(columns=rename_map)


def select_dim_demands(
    demands_df: pd.DataFrame, dim: str, question_id_col: str = "question_id"
) -> pd.DataFrame:
    """Extracts one dimension's (lower, upper) columns from a
    load_demands_multi_dim() DataFrame, renamed to lower/upper -- ready for
    build_arrays/fit_theta, matching load_demands_single_dim's output shape."""
    return demands_df[[question_id_col, f"{dim}_lower", f"{dim}_upper"]].rename(
        columns={f"{dim}_lower": "lower", f"{dim}_upper": "upper"}
    )


def load_outcomes(csv_path):
    """Reads a wide-format outcomes CSV: question_id + one
    `{model_name}_outcome` column per model."""
    return pd.read_csv(csv_path)


def build_arrays(demands_df, outcomes_df, model_name):
    """Inner-joins demands and outcomes on question_id and extracts the
    (b_l, b_u) demand array and the binary success array for one model.

    Returns (demands: (N, 2) ndarray of [lower, upper], success: (N,) ndarray).
    """
    df = pd.merge(demands_df, outcomes_df, on="question_id", how="inner")
    demands = df[["lower", "upper"]].values
    success = df[f"{model_name}_outcome"].values
    return demands, success
