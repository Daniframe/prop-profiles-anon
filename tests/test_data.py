import json

import pandas as pd
import pytest

from src.propensity.surfaces.data import (
    load_demands_single_dim as load_demands,
    load_demands_multi_dim,
    select_dim_demands,
    load_outcomes,
    build_arrays,
)


def test_build_arrays_joins_correctly(tmp_path):
    demands_path = tmp_path / "demands.jsonl"
    outcomes_path = tmp_path / "outcomes.csv"

    demand_rows = [
        {"question_id": 1, "propensity_lower": -2.0, "propensity_upper": 0.0},
        {"question_id": 2, "propensity_lower": -1.0, "propensity_upper": 1.0},
        {"question_id": 3, "propensity_lower": 0.0, "propensity_upper": 2.0},
    ]
    with open(demands_path, "w") as f:
        for row in demand_rows:
            f.write(json.dumps(row) + "\n")

    outcomes_df = pd.DataFrame({
        "question_id": [1, 2, 3],
        "model_a_outcome": [1, 0, 1],
        "model_b_outcome": [0, 1, 1],
    })
    outcomes_df.to_csv(outcomes_path, index=False)

    demands_df = load_demands(str(demands_path))
    outcomes_df_loaded = load_outcomes(str(outcomes_path))

    assert list(demands_df.columns) == ["question_id", "lower", "upper"]

    demands, success = build_arrays(demands_df, outcomes_df_loaded, "model_a")
    assert demands.shape == (3, 2)
    assert list(demands[:, 0]) == [-2.0, -1.0, 0.0]
    assert list(demands[:, 1]) == [0.0, 1.0, 2.0]
    assert list(success) == [1, 0, 1]

    _, success_b = build_arrays(demands_df, outcomes_df_loaded, "model_b")
    assert list(success_b) == [0, 1, 1]


def test_build_arrays_inner_join_drops_unmatched(tmp_path):
    demands_path = tmp_path / "demands.jsonl"
    outcomes_path = tmp_path / "outcomes.csv"

    with open(demands_path, "w") as f:
        f.write(json.dumps({"question_id": 1, "propensity_lower": -1.0, "propensity_upper": 1.0}) + "\n")
        f.write(json.dumps({"question_id": 2, "propensity_lower": -2.0, "propensity_upper": 2.0}) + "\n")

    pd.DataFrame({"question_id": [1], "model_a_outcome": [1]}).to_csv(outcomes_path, index=False)

    demands_df = load_demands(str(demands_path))
    outcomes_df = load_outcomes(str(outcomes_path))
    demands, success = build_arrays(demands_df, outcomes_df, "model_a")

    assert demands.shape == (1, 2)
    assert list(success) == [1]


def test_load_demands_single_dim_supports_csv(tmp_path):
    demands_path = tmp_path / "demands.csv"
    pd.DataFrame({
        "question_id": [1, 2],
        "propensity_lower": [-1.0, -2.0],
        "propensity_upper": [1.0, 2.0],
    }).to_csv(demands_path, index=False)

    demands_df = load_demands(str(demands_path))
    assert list(demands_df.columns) == ["question_id", "lower", "upper"]
    assert list(demands_df["lower"]) == [-1.0, -2.0]
    assert list(demands_df["upper"]) == [1.0, 2.0]


def test_load_demands_single_dim_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "demands.txt"
    path.write_text("question_id,propensity_lower,propensity_upper\n1,-1,1\n")
    with pytest.raises(ValueError):
        load_demands(str(path))


def test_load_demands_multi_dim_returns_one_dataframe_two_cols_per_dim(tmp_path):
    demands_path = tmp_path / "demands.jsonl"
    rows = [
        {"question_id": 1, "RA_l": -3, "RA_u": -1, "Ex_l": 0, "Ex_u": 2, "Ul_l": 1, "Ul_u": 3},
        {"question_id": 2, "RA_l": -2, "RA_u": 0, "Ex_l": -1, "Ex_u": 1, "Ul_l": 0, "Ul_u": 2},
    ]
    with open(demands_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    demands_df = load_demands_multi_dim(str(demands_path), dim_names=["RA", "Ex", "Ul"])

    assert list(demands_df.columns) == [
        "question_id",
        "RA_lower", "RA_upper",
        "Ex_lower", "Ex_upper",
        "Ul_lower", "Ul_upper",
    ]
    assert list(demands_df["RA_lower"]) == [-3, -2]
    assert list(demands_df["RA_upper"]) == [-1, 0]
    assert list(demands_df["Ex_lower"]) == [0, -1]
    assert list(demands_df["Ul_upper"]) == [3, 2]


def test_load_demands_multi_dim_custom_suffixes_and_csv(tmp_path):
    demands_path = tmp_path / "demands.csv"
    pd.DataFrame({
        "question_id": [1, 2],
        "RA.lower": [-3, -2],
        "RA.upper": [-1, 0],
    }).to_csv(demands_path, index=False)

    demands_df = load_demands_multi_dim(
        str(demands_path), dim_names=["RA"], lower_suffix=".lower", upper_suffix=".upper"
    )
    assert list(demands_df.columns) == ["question_id", "RA_lower", "RA_upper"]
    assert list(demands_df["RA_lower"]) == [-3, -2]
    assert list(demands_df["RA_upper"]) == [-1, 0]


def test_load_demands_multi_dim_raises_on_missing_dimension(tmp_path):
    demands_path = tmp_path / "demands.jsonl"
    with open(demands_path, "w") as f:
        f.write(json.dumps({"question_id": 1, "RA_l": -1, "RA_u": 1}) + "\n")

    with pytest.raises(ValueError):
        load_demands_multi_dim(str(demands_path), dim_names=["RA", "Ex"])


def test_select_dim_demands_extracts_one_dimension(tmp_path):
    demands_path = tmp_path / "demands.jsonl"
    rows = [
        {"question_id": 1, "RA_l": -3, "RA_u": -1, "Ex_l": 0, "Ex_u": 2},
        {"question_id": 2, "RA_l": -2, "RA_u": 0, "Ex_l": -1, "Ex_u": 1},
    ]
    with open(demands_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    multi = load_demands_multi_dim(str(demands_path), dim_names=["RA", "Ex"])

    ra = select_dim_demands(multi, "RA")
    assert list(ra.columns) == ["question_id", "lower", "upper"]
    assert list(ra["lower"]) == [-3, -2]
    assert list(ra["upper"]) == [-1, 0]

    ex = select_dim_demands(multi, "Ex")
    assert list(ex["lower"]) == [0, -1]
    assert list(ex["upper"]) == [2, 1]
