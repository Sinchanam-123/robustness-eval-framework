"""
Phase 5, Day 10 — insight generator.

Every statement is produced from computed thresholds against real numbers
in the robustness index table (which itself is computed from results.csv)
— never a hardcoded/templated string printed regardless of outcome. If a
statement can't trace back to a number in results.csv, it doesn't belong
here.

Run standalone: python -m insights.generator
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config
from evaluation.robustness_index import compute_robustness_index


def generate_insights(overall_df: pd.DataFrame = None, per_perturbation_df: pd.DataFrame = None) -> list[str]:
    if overall_df is None or per_perturbation_df is None:
        overall_df, per_perturbation_df, _ = compute_robustness_index()

    insights = []

    # Threshold check: flag any model whose overall index falls below the
    # configured "degrades significantly" cutoff.
    for _, row in overall_df.iterrows():
        if row["overall_robustness_index"] < config.ROBUSTNESS_INDEX_LOW_THRESHOLD:
            insights.append(
                f"{row['model']} degrades significantly under stress "
                f"(robustness index={row['overall_robustness_index']:.3f})"
            )

    # Best/worst overall, straight from the ranked table.
    ranked = overall_df.sort_values("overall_robustness_index", ascending=False).reset_index(drop=True)
    best = ranked.iloc[0]
    worst = ranked.iloc[-1]
    insights.append(
        f"{best['model']} is the most robust model overall "
        f"(index={best['overall_robustness_index']:.3f}); "
        f"{worst['model']} is the least robust "
        f"(index={worst['overall_robustness_index']:.3f})"
    )

    # Each model's single weakest perturbation type, from the per-perturbation table.
    for model in config.MODEL_NAMES:
        model_rows = per_perturbation_df[per_perturbation_df["model"] == model]
        if model_rows.empty:
            continue
        weakest = model_rows.loc[model_rows["robustness_index"].idxmin()]
        insights.append(
            f"{model}'s weakest perturbation is {weakest['perturbation']} "
            f"(retention={weakest['robustness_index']:.3f})"
        )

    return insights


if __name__ == "__main__":
    for line in generate_insights():
        print(line)
