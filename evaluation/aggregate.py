"""
Phase 4, Day 7 — aggregation.

Groups results.csv by (model, perturbation, severity) and computes mean +-
std across the 5 seeds for each metric. This is what turns "one run said X"
into "across 5 runs, X held with this much variance" — every later table,
chart, and API response reads from this aggregated view, not raw per-seed
rows.

Run standalone: python -m evaluation.aggregate
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import AGGREGATED_RESULTS_PATH, RESULTS_CSV_PATH

METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1"]


def load_results() -> pd.DataFrame:
    return pd.read_csv(RESULTS_CSV_PATH)


def aggregate_results(results_df: pd.DataFrame = None) -> pd.DataFrame:
    """One row per (model, perturbation, severity); {metric}_mean/{metric}_std columns."""
    if results_df is None:
        results_df = load_results()

    grouped = results_df.groupby(["model", "perturbation", "severity"])[METRIC_COLUMNS]
    agg = grouped.agg(["mean", "std"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    return agg.reset_index()


if __name__ == "__main__":
    aggregated_df = aggregate_results()
    AGGREGATED_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    aggregated_df.to_csv(AGGREGATED_RESULTS_PATH, index=False)
    print(aggregated_df.head(10).to_string(index=False))
    print(f"\n{len(aggregated_df)} rows written to {AGGREGATED_RESULTS_PATH}")
