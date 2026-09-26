import json

import pandas as pd
import pytest

from src.propensity.common.io import read_table, read_jsonl, write_jsonl


def test_read_table_reads_jsonl(tmp_path):
    path = tmp_path / "demands.jsonl"
    path.write_text('{"question_id": "q1", "lower": -1}\n{"question_id": "q2", "lower": 0}\n')

    df = read_table(path)

    assert list(df["question_id"]) == ["q1", "q2"]


def test_read_table_reads_csv(tmp_path):
    path = tmp_path / "outcomes.csv"
    path.write_text("question_id,model_outcome\nq1,1\nq2,0\n")

    df = read_table(path)

    assert list(df["model_outcome"]) == [1, 0]


def test_read_table_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "outcomes.txt"
    path.write_text("not a table")

    with pytest.raises(ValueError):
        read_table(path)


def test_write_jsonl_then_read_jsonl_round_trips(tmp_path):
    path = tmp_path / "nested" / "records.jsonl"
    records = [{"question_id": "q1", "lower": -1, "upper": 2}, {"question_id": "q2", "lower": 0, "upper": 3}]

    write_jsonl(records, path)
    result = list(read_jsonl(path))

    assert result == records


def test_write_jsonl_writes_one_json_object_per_line(tmp_path):
    path = tmp_path / "records.jsonl"
    write_jsonl([{"a": 1}, {"a": 2}], path)

    lines = path.read_text().splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0]) == {"a": 1}
    assert json.loads(lines[1]) == {"a": 2}
