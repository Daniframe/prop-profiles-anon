"""Rubric loading, resolved against the canonical rubrics/{code}/{code}_v1.md location."""

from pathlib import Path
from typing import Union


def rubric_path_for(code: str, rubrics_dir: Union[str, Path] = "rubrics") -> Path:
    """Returns the canonical path for a propensity dimension's rubric, e.g.
    rubric_path_for("RA") -> rubrics/RA/RA_v1.md."""
    return Path(rubrics_dir) / code / f"{code}_v1.md"


def load_rubric_text(path: Union[str, Path]) -> str:
    """Reads a rubric file's full text."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
