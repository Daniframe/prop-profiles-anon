"""Adapter for 'bring your own benchmark' external/real benchmarks that need
no synthetic generation (e.g. TimeMenatQA), only a column mapping onto the
canonical question_id/question_text/correct_answer shape used downstream by
annotation and inference.
"""

import pandas as pd

from .spec import BenchmarkSpec


def load_external_benchmark(spec: BenchmarkSpec) -> list[dict]:
    """Reads spec.params['source_path'] (CSV) and renames columns per
    spec.params['column_map'] ({source_column: canonical_column}); any
    columns listed in spec.params['extra_columns'] are carried through
    unrenamed (e.g. capability/difficulty metadata columns)."""
    p = spec.params
    df = pd.read_csv(p["source_path"])

    column_map = p["column_map"]
    extra_columns = p.get("extra_columns", [])
    selected = list(column_map) + extra_columns
    renamed = df[selected].rename(columns=column_map)
    return renamed.to_dict(orient="records")
