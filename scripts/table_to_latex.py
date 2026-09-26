"""Helper (not part of the package, not committed): reassembles
data/propsurfaces/props/FinalTableGoFAll.csv's already-formatted per-row
LaTeX fragments into the full gOFFull table, so it can be pasted straight
into a LaTeX editor and checked against the camera-ready table by eye.

Usage: python scripts/table_to_latex.py [--csv PATH] [--out PATH]
Prints to stdout if --out is omitted.
"""

import argparse
import os
import sys

import pandas as pd

DISPLAY_NAME = {
    "4o": "GPT-4o",
    "Nemo": "Nemo",
    "ds-r1-llama70": "DS-R1-Llama70B",
    "ds-r1-llama8": "DS-R1-Llama8B",
    "ds-r1-qwen32": "DS-R1-Qwen32B",
    "gemma3": "Gemma 3",
    "llama32": "Llama 3.2",
    "llama33": "Llama 3.3",
    "ministral-r": "Ministral 3-14B-R",
    "o1": "o1",
    "qwen3-4b-i": "Qwen 3-4B-I",
    "qwen3-4b-t": "Qwen 3-4B-T",
}

HEADER = r"""\begin{table}[!t]
\centering
\caption{Incited and obtained propensity levels with goodness-of-fit values}
\label{tab:gOFFull}
\setlength{\tabcolsep}{10pt}
\renewcommand{\arraystretch}{0.4}
\resizebox{1\columnwidth}{!}{%
\begin{tabular}{c r r r r r}
\toprule
\multirow{2}{*}{\textbf{Model}}
& \multicolumn{1}{c}{\textbf{Incited prop.}}
& \multicolumn{4}{c}{\textbf{Obtained prop. (Pseudo-$R^2$)}} \\
\cmidrule(lr){3-6}
& \multicolumn{1}{c}{\textbf{Level}}
& \multicolumn{1}{c}{\textbf{RvB.}}
& \multicolumn{1}{c}{\textbf{Risk Av.}}
& \multicolumn{1}{c}{\textbf{Introv.}}
& \multicolumn{1}{c}{\textbf{Ultracrep.}} \\
\midrule
"""

FOOTER = r"""\bottomrule
\end{tabular}
}
\end{table}
"""


def highlight_unprompted(row_text):
    """The CSV's unprompted row is plain text; the camera-ready table
    prefixes every cell in that row with \\cellcolor{srow!60}."""
    cells = row_text.split("&")
    return " & ".join(
        f"\\cellcolor{{srow!60}} {cell.strip()}" if cell.strip() else cell
        for cell in cells
    )


def build_table(df):
    lines = [HEADER.rstrip("\n")]
    models = df["Model"].unique().tolist()
    for i, model in enumerate(models):
        sub = df[df["Model"] == model]
        display = DISPLAY_NAME.get(model, model)
        n = len(sub)
        for j, (_, r) in enumerate(sub.iterrows()):
            text = r["Final text"]
            if str(r["Incited propensity"]).strip().lower() == "unprompted":
                text = highlight_unprompted(text)
            if j == 0:
                lines.append(f"\\multirow{{{n}}}{{*}}{{{display}}} {text}")
            else:
                lines.append(f" {text}")
        if i < len(models) - 1:
            lines.append("\\midrule")
        else:
            lines.append(FOOTER.rstrip("\n"))
    return "\n".join(lines) + "\n"


def main():
    default_csv = os.path.join(os.path.dirname(__file__), "..", "data", "propsurfaces", "props", "FinalTableGoFAll.csv")
    parser = argparse.ArgumentParser(description="Reassemble FinalTableGoFAll.csv into the full gOFFull LaTeX table")
    parser.add_argument("--csv", type=str, default=default_csv)
    parser.add_argument("--out", type=str, default=None, help="write to this file instead of stdout")
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"Incited propensity": str})
    table = build_table(df)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(table)


if __name__ == "__main__":
    main()
