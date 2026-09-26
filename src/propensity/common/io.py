"""Shared JSONL/CSV read-write helpers used across sectors."""

import json
from pathlib import Path
from typing import Any, Iterator, Union

import pandas as pd


def read_table(path: Union[str, Path]) -> pd.DataFrame:
    """Reads a JSONL or CSV file into a DataFrame, format inferred from its extension."""
    path = Path(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file extension '{path.suffix}' (expected .jsonl or .csv)")


def read_jsonl(path: Union[str, Path]) -> Iterator[dict]:
    """Yields one dict per line from a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: list[dict[str, Any]], path: Union[str, Path]) -> None:
    """Writes a list of dicts to a JSONL file, one record per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
