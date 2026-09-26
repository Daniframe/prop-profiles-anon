"""General CLI entry point for Sector 5's predictability module: given an
already-joined table, compare named feature-set configurations at
predicting a binary label via 10-fold-CV Random Forest AUROC.

Dataset-agnostic -- it does no joining/column-cleanup of its own, unlike
scripts/compare_assessors_TimeMenatQA.py, which is the TimeMenatQA-specific
driver that builds such a table and calls into the same underlying
src/propensity/predictability module directly.

Usage:
    python scripts/run_assessors.py --table data.csv --label-column outcome \
        --feature-set baseline=col1,col2 \
        --feature-set augmented=col1,col2,col3,col4 \
        --out comparison.csv
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.propensity.common.io import read_table
from src.propensity.predictability import compare_feature_sets


def parse_feature_set(spec: str):
    """Parses one --feature-set "name=col1,col2,col3" argument."""
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--feature-set must be 'name=col1,col2,...', got: {spec!r}")
    name, columns = spec.split("=", 1)
    columns = [c.strip() for c in columns.split(",") if c.strip()]
    if not columns:
        raise argparse.ArgumentTypeError(f"--feature-set '{name}' has no columns: {spec!r}")
    return name, columns


def main():
    parser = argparse.ArgumentParser(
        description="Compare named feature-set configurations at predicting a binary label (10-fold-CV RF AUROC)"
    )
    parser.add_argument("--table", type=str, required=True, help="CSV or JSONL file with all feature and label columns already joined")
    parser.add_argument("--label-column", type=str, required=True)
    parser.add_argument("--feature-set", action="append", required=True, dest="feature_sets",
                         help="'name=col1,col2,...' -- repeat for each named feature set to compare")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    feature_sets = dict(parse_feature_set(spec) for spec in args.feature_sets)

    df = read_table(args.table)
    result = compare_feature_sets(df, feature_sets, args.label_column, seed=args.seed)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Compared {len(feature_sets)} feature set(s), wrote {args.out}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
