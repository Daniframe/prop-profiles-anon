"""CLI entry point for Sector 5's TimeMenatQA predictability comparison:
does adding propensity-demand features to capability-demand features
improve a Random Forest assessor's AUROC at predicting per-instance model
correctness?

This script is the TimeMenatQA-specific *driver* -- joining ready/annotated/
outcomes on question_id, resolving a couple of known duplicate/inconsistent
model columns, and naming the three feature-set configurations the paper
compares. The reusable logic (generic feature selection, 10-fold CV
evaluation, side-by-side comparison) lives in
src/propensity/predictability/, dataset-agnostic -- see
scripts/run_assessors.py for a general CLI over that same module that
takes any already-joined table and caller-named feature sets.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

from src.propensity.common.io import read_table
from src.propensity.predictability import compare_feature_sets

# The 19-column capability-demand slice (ADeLe/DeLeAn dimensions), matching
# the original assessors.py's feature slice exactly --
# includes UG (constant across all TimeMenatQA items, harmless to an RF
# split) even though the paper's prose states 18 dims and excludes it.
CAPABILITY_COLUMNS = [
    "UG", "AS", "CEc", "CEe", "CL", "KNn", "KNa", "KNc", "KNf", "KNs",
    "MCt", "MCu", "MCr", "MS", "QLq", "QLl", "SNs", "VO", "AT",
]

PROPENSITY_CODES = ["BR", "RA", "Ex", "Ul"]
LEVELS = ["-2", "0", "+2"]

# TimeMenatQA_complete.csv has two known data-quality issues, confirmed by
# direct comparison: "gpt-4o_*" is a byte-for-byte duplicate of "4o_*" (drop
# it), and "ministral_*"/"ministral-r_*" are genuinely different inference
# runs (69-90% agreement) -- keep "ministral-r" only, matching the model
# shortcut used everywhere else in this repo.
DROP_MODEL_PREFIXES = ["gpt-4o", "ministral"]


def load_joined_table(base_dir: str) -> pd.DataFrame:
    ready = read_table(os.path.join(base_dir, "data", "benchmarks", "ready", "TimeMenatQA", "TimeMenatQA.jsonl"))
    df = ready[["question_id"] + CAPABILITY_COLUMNS]

    for code in PROPENSITY_CODES:
        annotated = read_table(os.path.join(
            base_dir, "data", "benchmarks", "annotated", "TimeMenatQA", f"TimeMenatQA_{code}_annotations_GPT-4.1.jsonl"
        ))
        annotated = annotated[["question_id", "propensity_lower", "propensity_upper"]].rename(columns={
            "propensity_lower": f"{code}_lower", "propensity_upper": f"{code}_upper",
        })
        df = df.merge(annotated, on="question_id", how="inner")

    outcomes = read_table(os.path.join(base_dir, "data", "inference", "outcomes", "TimeMenatQA", "TimeMenatQA_complete.csv"))
    df = df.merge(outcomes, on="question_id", how="inner")

    drop_cols = [c for c in df.columns if any(c.startswith(f"{p}_") for p in DROP_MODEL_PREFIXES)]
    df = df.drop(columns=drop_cols)
    return df


def resolve_models(df: pd.DataFrame) -> list:
    outcome_cols = [c for c in df.columns if c.endswith("_outcome")]
    models = sorted({c.split("_Ul_")[0] for c in outcome_cols})
    return models


def feature_sets_for(code: str) -> dict:
    all_prop_cols = [col for c in PROPENSITY_CODES for col in (f"{c}_lower", f"{c}_upper")]
    return {
        "capabilities_only": CAPABILITY_COLUMNS,
        "capabilities_plus_ultracrep": CAPABILITY_COLUMNS + [f"{code}_lower", f"{code}_upper"],
        "capabilities_plus_all_props": CAPABILITY_COLUMNS + all_prop_cols,
    }


DEFAULT_OUT = os.path.join("data", "propsurfaces", "predictability", "TimeMenatQA_assessor_table.csv")


def build_comparison_table(df: pd.DataFrame, models: list) -> pd.DataFrame:
    """Trains all 27 (model, level) x 3-feature-set-config assessors and
    returns the wide per-subject AUROC table -- the reusable core of this
    script's training step, factored out so other scripts (e.g.
    population_significance_TimeMenatQA.py) can reuse it without
    duplicating the loop or re-running scripts/compare_assessors_TimeMenatQA.py
    as a subprocess.
    """
    feature_sets = feature_sets_for("Ul")
    rows = []
    for model in models:
        for level in LEVELS:
            label_column = f"{model}_Ul_{level}_outcome"
            if label_column not in df.columns:
                continue
            comparison = compare_feature_sets(df, feature_sets, label_column)
            for _, r in comparison.iterrows():
                rows.append({"Model": model, "Bias": level, "FeatureSet": r["feature_set"], "AUROC": r["auroc"]})

    long_table = pd.DataFrame(rows)
    wide_table = long_table.pivot(index=["Model", "Bias"], columns="FeatureSet", values="AUROC").reset_index()
    return wide_table[["Model", "Bias", "capabilities_only", "capabilities_plus_ultracrep", "capabilities_plus_all_props"]]


def main():
    parser = argparse.ArgumentParser(
        description="Compare capability-only vs. capability+propensity Random Forest assessors on TimeMenatQA"
    )
    parser.add_argument("--out", type=str, default=DEFAULT_OUT)
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    df = load_joined_table(base_dir)
    models = resolve_models(df)
    wide_table = build_comparison_table(df, models)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wide_table.to_csv(args.out, index=False)
    print(f"Evaluated {len(models)} models x {len(LEVELS)} levels, wrote {len(wide_table)} rows -> {args.out}")


if __name__ == "__main__":
    main()
