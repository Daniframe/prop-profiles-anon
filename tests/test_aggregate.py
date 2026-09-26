import numpy as np
import pandas as pd

from src.propensity.surfaces.model import two_sided_sigma
from src.propensity.surfaces.aggregate import derive_propensity_table


def _simulate_outcomes_df(theta_true, n_items=150, seed=0):
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-3, 3, n_items)
    half_widths = rng.uniform(0.5, 1.5, n_items)
    lower = np.round(centers - half_widths).astype(int)
    upper = np.round(centers + half_widths).astype(int)
    probs = np.array([two_sided_sigma(theta_true, lo, hi, 1.0, 1.0) for lo, hi in zip(lower, upper)])
    success = (rng.random(n_items) < probs).astype(int)
    return lower, upper, success


def _fixture_dfs():
    question_ids = [f"Q_{i}" for i in range(150)]
    lower, upper, success_m1 = _simulate_outcomes_df(theta_true=-1.0, seed=1)
    _, _, success_0 = _simulate_outcomes_df(theta_true=0.0, seed=2)

    demands_df = pd.DataFrame({"question_id": question_ids, "lower": lower, "upper": upper})
    outcomes_df = pd.DataFrame({
        "question_id": question_ids,
        "modelA_RA_-1_outcome": success_m1,
        "modelA_RA_0_outcome": success_0,
        "modelA_RA_outcome": success_0,  # "unprompted" bare column
    })
    return demands_df, outcomes_df


def test_derive_propensity_table_fits_expected_columns():
    demands_df, outcomes_df = _fixture_dfs()
    table = derive_propensity_table(demands_df, outcomes_df, models=["modelA"], code="RA", levels=["-1", "0"])

    assert list(table.columns) == [
        "Model", "Incited propensity", "Obtained propensity", "Lower 95CI",
        "Upper 95CI", "Converges", "ReferenceLL", "EstimationLL",
    ]
    assert len(table) == 2
    assert set(table["Model"]) == {"modelA"}
    assert set(table["Incited propensity"]) == {-1.0, 0.0}


def test_derive_propensity_table_recovers_true_theta_sign():
    demands_df, outcomes_df = _fixture_dfs()
    table = derive_propensity_table(demands_df, outcomes_df, models=["modelA"], code="RA", levels=["-1"])

    row = table.iloc[0]
    assert row["Obtained propensity"] < 0  # simulated with theta_true=-1.0


def test_derive_propensity_table_handles_unprompted_none_level():
    demands_df, outcomes_df = _fixture_dfs()
    table = derive_propensity_table(demands_df, outcomes_df, models=["modelA"], code="RA", levels=[None])

    assert len(table) == 1
    assert pd.isna(table.iloc[0]["Incited propensity"])


def test_derive_propensity_table_skips_missing_model_level_columns():
    demands_df, outcomes_df = _fixture_dfs()
    table = derive_propensity_table(
        demands_df, outcomes_df, models=["modelA", "modelB_not_present"], code="RA", levels=["-1"]
    )

    assert set(table["Model"]) == {"modelA"}


def test_derive_propensity_table_skips_levels_with_no_matching_column():
    demands_df, outcomes_df = _fixture_dfs()
    table = derive_propensity_table(demands_df, outcomes_df, models=["modelA"], code="RA", levels=["-1", "+3"])

    assert set(table["Incited propensity"]) == {-1.0}  # "+3" column doesn't exist, silently skipped


def test_derive_propensity_table_robust_true_adds_diagnostic_columns():
    demands_df, outcomes_df = _fixture_dfs()
    table = derive_propensity_table(
        demands_df, outcomes_df, models=["modelA"], code="RA", levels=["-1", "0"],
        robust=True, max_retries=10, patience=2,
    )

    assert {"NAttempts", "NConverged", "RestartThetaStd"} <= set(table.columns)
    assert (table["NAttempts"] >= 1).all()
