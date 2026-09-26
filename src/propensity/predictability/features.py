"""Generic feature/label table construction for predictability assessors.

Deliberately dataset-agnostic: the caller names whichever columns constitute
a feature set (capability demands, propensity demand bounds, or anything
else already joined into one dataframe) -- this module has no built-in
notion of what a "capability" or "propensity" column is.
"""

from typing import List, Tuple

import pandas as pd


def build_feature_table(
    df: pd.DataFrame, feature_columns: List[str], label_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Selects feature_columns and label_column from df, coerces them to
    numeric, and drops any row with a NaN in a selected column or the label.

    Returns (X, y) ready for a classifier -- X has exactly feature_columns
    as its columns, y is the coerced label_column, both re-indexed from 0.
    """
    selected = df[feature_columns + [label_column]].apply(pd.to_numeric, errors="coerce")
    selected = selected.dropna()
    X = selected[feature_columns].reset_index(drop=True)
    y = selected[label_column].reset_index(drop=True)
    return X, y
