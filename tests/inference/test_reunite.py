import json

import pandas as pd

from src.propensity.inference.reunite import (
    load_results,
    exact_match,
    judgment_field,
    score_responses,
    build_outcomes_table,
)


def test_load_results_reads_run_inference_output_shape(tmp_path):
    path = tmp_path / "results.json"
    payload = {"meta": {"model": "gpt-4o"}, "results": {"0": {"total": 1, "responses": [{"question_id": "q1"}]}}}
    path.write_text(json.dumps(payload))

    loaded = load_results(path)
    assert loaded == payload


def test_exact_match_true_when_answer_starts_with_gold():
    response = {"final_answer": "A - the safe option", "correct_answer": "A - the safe option"}
    assert exact_match(response) is True


def test_exact_match_false_on_mismatch():
    response = {"final_answer": "B - the risky option", "correct_answer": "A - the safe option"}
    assert exact_match(response) is False


def test_exact_match_false_on_missing_fields():
    assert exact_match({}) is False


def test_judgment_field_reads_boolean():
    assert judgment_field({"judgment": True}) is True
    assert judgment_field({"judgment": False}) is False
    assert judgment_field({}) is False


def test_score_responses_applies_default_exact_match():
    responses = [
        {"question_id": "q1", "final_answer": "A", "correct_answer": "A"},
        {"question_id": "q2", "final_answer": "B", "correct_answer": "A"},
    ]
    scored = score_responses(responses)
    assert scored == {"q1": 1, "q2": 0}


def test_score_responses_accepts_custom_predicate():
    responses = [{"question_id": "q1", "judgment": True}, {"question_id": "q2", "judgment": False}]
    scored = score_responses(responses, is_correct=judgment_field)
    assert scored == {"q1": 1, "q2": 0}


def test_build_outcomes_table_pivots_multiple_columns():
    columns = {
        "4o_RA_-1_outcome": {"RA_0": 1, "RA_1": 0},
        "4o_RA_0_outcome": {"RA_0": 1, "RA_1": 1},
    }
    table = build_outcomes_table(columns)

    assert list(table.columns) == ["question_id", "4o_RA_-1_outcome", "4o_RA_0_outcome"]
    assert table.set_index("question_id").loc["RA_0"].tolist() == [1, 1]
    assert table.set_index("question_id").loc["RA_1"].tolist() == [0, 1]


def test_build_outcomes_table_leaves_missing_entries_as_nan():
    columns = {
        "col_a": {"RA_0": 1, "RA_1": 0},
        "col_b": {"RA_0": 1},  # RA_1 missing for this column
    }
    table = build_outcomes_table(columns).set_index("question_id")

    assert table.loc["RA_1", "col_a"] == 0
    assert pd.isna(table.loc["RA_1", "col_b"])
