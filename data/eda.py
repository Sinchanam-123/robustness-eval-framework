"""
Phase 1, Day 2 — exploratory data analysis on the Adult Census Income dataset.

Run standalone: python -m data.eda
Prints nulls, dtypes, class balance, and feature ranges/std, and writes the
same findings to data/feature_stats.md so later stress-test severity scaling
has a written reference for each feature's scale.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import TARGET_COLUMN, FEATURE_STATS_PATH
from data.preprocessing import load_raw, NUMERIC_COLUMNS


def run_eda(df: pd.DataFrame) -> str:
    lines = []
    lines.append("# Adult Census Income — EDA notes (Phase 1, Day 2)\n")

    lines.append("## Shape")
    lines.append(f"- rows: {df.shape[0]}, columns: {df.shape[1]}\n")

    lines.append("## Dtypes")
    for col, dtype in df.dtypes.items():
        lines.append(f"- {col}: {dtype}")
    lines.append("")

    lines.append("## Nulls (encoded as \"?\" in raw data)")
    null_counts = df.isna().sum()
    for col, count in null_counts.items():
        if count > 0:
            pct = 100 * count / len(df)
            lines.append(f"- {col}: {count} ({pct:.2f}%)")
    if null_counts.sum() == 0:
        lines.append("- none")
    lines.append("")

    lines.append(f"## Class balance ({TARGET_COLUMN})")
    balance = df[TARGET_COLUMN].value_counts(normalize=True)
    for label, frac in balance.items():
        lines.append(f"- {label}: {frac:.4f}")
    lines.append(
        "\nNote: real class imbalance (~76%/24%) — this is why accuracy alone "
        "is not sufficient and precision/recall/F1 are tracked too.\n"
    )

    lines.append("## Numeric feature ranges (min / max / mean / std)")
    lines.append(
        "These std values are what stress-test severity scaling multiplies "
        "against (e.g. `add_noise` uses `X.std(axis=0) * severity`) — recorded "
        "here so severity can be sanity-checked against real feature scale.\n"
    )
    lines.append("| feature | min | max | mean | std |")
    lines.append("|---|---|---|---|---|")
    for col in NUMERIC_COLUMNS:
        s = df[col]
        lines.append(
            f"| {col} | {s.min():.2f} | {s.max():.2f} | {s.mean():.2f} | {s.std():.2f} |"
        )
    lines.append("")

    lines.append("## Categorical feature cardinality")
    for col in df.columns:
        if col not in NUMERIC_COLUMNS and col != TARGET_COLUMN:
            lines.append(f"- {col}: {df[col].nunique()} unique values")
    lines.append("")

    return "\n".join(lines)


def main():
    df = load_raw()
    report = run_eda(df)
    print(report)
    FEATURE_STATS_PATH.write_text(report, encoding="utf-8")
    print(f"\nWritten to {FEATURE_STATS_PATH}")


if __name__ == "__main__":
    main()
