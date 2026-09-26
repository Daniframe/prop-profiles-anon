"""CLI entry point for Sector 5's TimeMenatQA population-level significance
test: trains the 27 per-(model, level) assessors across the three feature-set
configurations (reusing compare_assessors_TimeMenatQA.py's exact training
logic, no duplication) and tests, pairwise, whether the AUROC differences
between configurations are significant across all 27 subjects.

Unlike a per-subject test (does propensity help for THIS specific model and
level), this asks whether the pattern generalises across subjects -- the
question the paper's own closing claim ("this pattern is observed
independent of incitation level... generalise across models") is actually
about. Three tests per pair, all cheap (they only need the 27 already-
computed subject-level AUROCs, no refitting):

- population_paired_test (Wilcoxon signed-rank on the 27 deltas)
- population_win_rate_test (binomial sign test on the win count)
- population_bootstrap_ci (percentile CI on the mean delta)

See src/propensity/predictability/evaluation.py for each test's exact
methodology and assumptions.
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

from compare_assessors_TimeMenatQA import load_joined_table, resolve_models, build_comparison_table
from src.propensity.predictability import population_paired_test, population_win_rate_test, population_bootstrap_ci

PAIRS = [
    ("capabilities_only", "capabilities_plus_ultracrep"),
    ("capabilities_only", "capabilities_plus_all_props"),
    ("capabilities_plus_ultracrep", "capabilities_plus_all_props"),
]

DEFAULT_ASSESSOR_TABLE_OUT = os.path.join("data", "propsurfaces", "predictability", "TimeMenatQA_assessor_table.csv")
DEFAULT_SIGNIFICANCE_OUT = os.path.join("data", "propsurfaces", "predictability", "TimeMenatQA_population_significance.csv")


def population_significance(assessor_table: pd.DataFrame, seed: int = 42, n_resamples: int = 10000) -> pd.DataFrame:
    rows = []
    for a, b in PAIRS:
        deltas = (assessor_table[b] - assessor_table[a]).tolist()
        wins = sum(d > 0 for d in deltas)
        n = len(deltas)

        paired = population_paired_test(deltas, test="wilcoxon")
        winrate = population_win_rate_test(wins, n)
        ci = population_bootstrap_ci(deltas, seed=seed, n_resamples=n_resamples)

        rows.append({
            "baseline (A)": a, "compared (B)": b,
            "mean_delta(B-A)": paired["mean_delta"], "median_delta": paired["median_delta"],
            "wilcoxon_stat": paired["statistic"], "wilcoxon_p": paired["pvalue"],
            "wins_B_over_A": f"{wins}/{n}", "winrate_p": winrate["pvalue"],
            "bootstrap_ci_lower": ci["ci_lower"], "bootstrap_ci_upper": ci["ci_upper"],
        })
    return pd.DataFrame(rows)


def print_highlight(significance: pd.DataFrame) -> None:
    """Flags which propensity-augmented configuration(s) significantly beat
    capabilities_only -- the 95% bootstrap CI on the delta must exclude
    zero on the positive side."""
    print("\n=== Highlight: propensity configs vs. capabilities_only ===")
    baseline_rows = significance[significance["baseline (A)"] == "capabilities_only"]
    for _, row in baseline_rows.iterrows():
        significant = row["bootstrap_ci_lower"] > 0
        verdict = "SIGNIFICANT" if significant else "not significant"
        print(f"  {row['compared (B)']}: mean delta={row['mean_delta(B-A)']:.4f}, "
              f"wilcoxon p={row['wilcoxon_p']:.4g}, win-rate p={row['winrate_p']:.4g}, "
              f"95% CI=[{row['bootstrap_ci_lower']:.4f}, {row['bootstrap_ci_upper']:.4f}] -> {verdict}")


def main():
    parser = argparse.ArgumentParser(
        description="Train TimeMenatQA per-subject assessors and test population-level significance of the AUROC differences, pairwise"
    )
    parser.add_argument("--assessor-table-out", type=str, default=DEFAULT_ASSESSOR_TABLE_OUT)
    parser.add_argument("--significance-out", type=str, default=DEFAULT_SIGNIFICANCE_OUT)
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_dir = os.path.join(os.path.dirname(__file__), "..")
    df = load_joined_table(base_dir)
    models = resolve_models(df)

    print(f"Training {len(models)} models x 3 levels x 3 feature-set configurations...")
    assessor_table = build_comparison_table(df, models)
    os.makedirs(os.path.dirname(args.assessor_table_out) or ".", exist_ok=True)
    assessor_table.to_csv(args.assessor_table_out, index=False)
    print(assessor_table.to_string(index=False))

    print("\nRunning population-level significance tests (pairwise, all 3 configs)...")
    significance = population_significance(assessor_table, seed=args.seed, n_resamples=args.n_resamples)
    os.makedirs(os.path.dirname(args.significance_out) or ".", exist_ok=True)
    significance.to_csv(args.significance_out, index=False)
    print(significance.to_string(index=False))

    print_highlight(significance)
    print(f"\nWrote {args.assessor_table_out} and {args.significance_out}")


if __name__ == "__main__":
    main()
