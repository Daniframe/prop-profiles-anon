"""Helpers for turning raw inference results into outcome columns/tables --
deliberately a small toolkit, not an opinionated pipeline. The original
reunite_results.py/reunite_zero.py hardcoded a directory walk
over a fixed dataset x model x level matrix and a fixed column-naming
convention; that's a decision specific to how *you* want to organize a run,
not something this module should dictate. What it gives you:

  - score_responses(): turn one prompt's responses into a {question_id: 0/1} column,
    given whatever correctness rule fits that benchmark (exact match, judge
    output, or your own callable).
  - build_outcomes_table(): pivot any number of such columns into one wide
    DataFrame keyed by question_id -- the same shape as the existing
    data/inference/outcomes/RA/RA_complete.csv, but you choose the column
    names, which models/levels to include, and how many results files to
    pull from.

Write your own script/notebook on top of these for your specific run layout
(see docs/sector3_inference.md for a worked example).
"""

import json
from pathlib import Path
from typing import Callable, Union

import pandas as pd


def load_results(path: Union[str, Path]) -> dict:
    """Loads a results JSON as written by scripts/run_inference.py
    ({"meta": {...}, "results": {prompt_name: {"total": N, "responses": [...]}}})."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def exact_match(response: dict, answer_field: str = "final_answer", gold_field: str = "correct_answer") -> bool:
    """A common-case correctness rule for multiple choice: does the
    extracted final answer start with the correct answer's text (or vice
    versa)? Only a sensible default for benchmarks shaped like Sector 1's
    MCQ output -- for anything else (open-ended answers, custom grading),
    write your own callable and pass it to score_responses instead."""
    answer = str(response.get(answer_field, "")).strip().lower()
    gold = str(response.get(gold_field, "")).strip().lower()
    if not answer or not gold:
        return False
    return answer.startswith(gold) or gold.startswith(answer)


def judgment_field(response: dict, field: str = "judgment") -> bool:
    """Correctness rule for items already graded by inference.judge (or
    anything else that wrote a boolean `judgment` field onto each response)."""
    return bool(response.get(field, False))


def score_responses(
    responses: list[dict],
    is_correct: Callable[[dict], bool] = exact_match,
    question_id_field: str = "question_id",
) -> dict[str, int]:
    """Scores one prompt's responses (e.g. results["results"][prompt_name]["responses"])
    into {question_id: 1 or 0} using `is_correct`."""
    return {r[question_id_field]: int(is_correct(r)) for r in responses}


def build_outcomes_table(
    columns: dict[str, dict[str, int]],
    question_id_col: str = "question_id",
) -> pd.DataFrame:
    """Pivots any number of {question_id: 0/1} columns (as produced by
    score_responses, one per model/incitement-level/whatever you're
    comparing) into a single wide DataFrame -- one row per question_id seen
    in any column, one column per key in `columns`. Missing entries (a
    question_id present in one column's source data but not another's) are
    left as NaN rather than silently dropped or zero-filled.

    columns: {column_name: {question_id: 0/1}}, e.g.
        {"4o_RA_-1_outcome": score_responses(...), "4o_RA_0_outcome": score_responses(...)}
    """
    series = {name: pd.Series(outcomes, name=name) for name, outcomes in columns.items()}
    df = pd.concat(series.values(), axis=1)
    df.index.name = question_id_col
    return df.reset_index()
