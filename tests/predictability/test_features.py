import numpy as np
import pandas as pd

from src.propensity.predictability.features import build_feature_table


def _fixture_df():
    return pd.DataFrame({
        "a": [1, 2, 3, 4, np.nan],
        "b": [10, 20, 30, np.nan, 50],
        "unused": ["x", "y", "z", "w", "v"],
        "label": [1, 0, 1, 0, 1],
    })


def test_build_feature_table_selects_only_requested_columns():
    df = _fixture_df()
    X, y = build_feature_table(df, ["a", "b"], "label")
    assert list(X.columns) == ["a", "b"]
    assert "unused" not in X.columns


def test_build_feature_table_drops_rows_with_nan_in_features():
    df = _fixture_df()
    X, y = build_feature_table(df, ["a", "b"], "label")
    # rows 3 (b=NaN) and 4 (a=NaN) should be dropped
    assert len(X) == 3
    assert len(y) == 3


def test_build_feature_table_drops_rows_with_nan_in_label():
    df = _fixture_df()
    df.loc[0, "label"] = np.nan
    X, y = build_feature_table(df, ["a"], "label")
    assert len(X) == len(y)
    assert not y.isna().any()


def test_build_feature_table_reindexes_from_zero():
    df = _fixture_df()
    X, y = build_feature_table(df, ["a", "b"], "label")
    assert list(X.index) == list(range(len(X)))
    assert list(y.index) == list(range(len(y)))


def test_build_feature_table_coerces_string_numerics():
    df = pd.DataFrame({"a": ["1.0", "2.0", "3.0"], "label": [1, 0, 1]})
    X, y = build_feature_table(df, ["a"], "label")
    assert X["a"].dtype.kind == "f"
    assert len(X) == 3
