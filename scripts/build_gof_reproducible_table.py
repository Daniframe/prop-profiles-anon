"""Helper (not part of the package, not committed): builds
data/propsurfaces/props/GoFAllReproducible.csv from this pipeline's own
Derived_propensities_{RA,BR,Ex,Ul}.csv, in the same (Model, Incited
propensity, Final text) schema as the original FinalTableGoFAll.csv --
so it can be fed straight into table_to_latex.py for a side-by-side
LaTeX comparison against the camera-ready table.

Usage: python scripts/build_gof_reproducible_table.py [--props-dir PATH] [--out PATH]
"""

import argparse
import os

import pandas as pd

# Model order matches FinalTableGoFAll.csv's row order (not this pipeline's own
# internal MODELS list order) so the two tables line up model-for-model.
MODEL_ORDER = [
    "4o", "Nemo", "ds-r1-llama70", "ds-r1-llama8", "ds-r1-qwen32", "gemma3",
    "llama32", "llama33", "ministral-r", "o1", "qwen3-4b-i", "qwen3-4b-t",
]
LEVEL_ORDER = [-3, -2, -1, 0, 1, 2, 3, None]

# Dimension order in the table is RvB, Risk Av., Introv., Ultracrep. -- our
# short codes for those are BR, RA, Ex, Ul respectively.
DIM_ORDER = ["BR", "RA", "Ex", "Ul"]


def load_dim(props_dir, code):
    df = pd.read_csv(os.path.join(props_dir, f"Derived_propensities_{code}.csv"))
    df["se"] = (df["Upper 95CI"] - df["Lower 95CI"]) / (2 * 1.96)
    df["pseudo_r2"] = 1 - (df["EstimationLL"] / df["ReferenceLL"])
    # NB: building a "level_key" column via .apply() would get silently
    # coerced back to float64 (turning the None sentinel into NaN again,
    # indistinguishable from itself) -- keyed directly off each row instead.
    result = {}
    for _, r in df.iterrows():
        level = r["Incited propensity"]
        level_key = None if pd.isna(level) else int(level)
        result[(r["Model"], level_key)] = r
    return result


def fmt_level(level):
    return "Unprompted" if level is None else f"{level:.2f}"


def fmt_cell(row):
    return f"${row['Obtained propensity']:.2f} \\pm {row['se']:.2f}$ ({row['pseudo_r2']:.2f})"


def build_rows(props_dir):
    dims = {code: load_dim(props_dir, code) for code in DIM_ORDER}
    rows = []
    for model in MODEL_ORDER:
        for level in LEVEL_ORDER:
            cells = []
            missing = False
            for code in DIM_ORDER:
                row = dims[code].get((model, level))
                if row is None:
                    missing = True
                    break
                cells.append(fmt_cell(row))
            if missing:
                print(f"skipping {model} @ {level}: missing in at least one dimension")
                continue
            text = f"& {fmt_level(level)} & " + " & ".join(cells) + r" \\"
            rows.append({
                "Model": model,
                "Incited propensity": fmt_level(level) if level is None else f"{level:.2f}",
                "Final text": text,
            })
    return rows


def main():
    default_props_dir = os.path.join(os.path.dirname(__file__), "..", "data", "propsurfaces", "props")
    parser = argparse.ArgumentParser(description="Build GoFAllReproducible.csv from this pipeline's own Derived_propensities_*.csv")
    parser.add_argument("--props-dir", type=str, default=default_props_dir)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    out = args.out or os.path.join(args.props_dir, "GoFAllReproducible.csv")

    rows = build_rows(args.props_dir)
    pd.DataFrame(rows, columns=["Model", "Incited propensity", "Final text"]).to_csv(out, index=False)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
