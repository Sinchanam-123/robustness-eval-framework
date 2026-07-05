"""
Phase 4, Day 8 — the robustness index.

    retention(severity) = metric(severity) / metric(severity=0)
    robustness_index = mean(retention across tested severities)

metric(severity=0) is the clean-test baseline from Phase 2
(models/baseline_metrics.csv) — the anchor every retention number is
relative to. One robustness index is computed per (model, perturbation),
then averaged across perturbation types for one overall score per model.

Also runs a paired t-test (across seeds) between the top two models'
per-seed robustness index, as a real statistical claim instead of just
"model A's number is bigger."

Run standalone: python -m evaluation.robustness_index
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scipy import stats

import config
from evaluation.aggregate import load_results


def compute_retention(results_df: pd.DataFrame, baseline_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """retention per (model, perturbation, severity, seed) row."""
    baseline_lookup = baseline_df.set_index("model")[metric]
    retention_df = results_df.copy()
    retention_df["baseline_metric"] = retention_df["model"].map(baseline_lookup)
    retention_df["retention"] = retention_df[metric] / retention_df["baseline_metric"]
    return retention_df


def compute_robustness_index(metric: str = None):
    """Returns (overall_df, per_perturbation_df, retention_df)."""
    metric = metric or config.ROBUSTNESS_METRIC
    results_df = load_results()
    baseline_df = pd.read_csv(config.BASELINE_METRICS_PATH)

    retention_df = compute_retention(results_df, baseline_df, metric)

    # mean retention across the 5 seeds -> one value per (model, perturbation, severity)
    per_severity = (
        retention_df.groupby(["model", "perturbation", "severity"])["retention"]
        .mean()
        .reset_index()
    )

    # robustness_index per (model, perturbation) = mean retention across severities
    per_perturbation = (
        per_severity.groupby(["model", "perturbation"])["retention"]
        .mean()
        .reset_index()
        .rename(columns={"retention": "robustness_index"})
    )

    # overall robustness_index per model = mean across perturbation types
    overall = (
        per_perturbation.groupby("model")["robustness_index"]
        .mean()
        .reset_index()
        .rename(columns={"robustness_index": "overall_robustness_index"})
        .sort_values("overall_robustness_index", ascending=False)
    )

    return overall, per_perturbation, retention_df


def paired_ttest_top_two(overall_df: pd.DataFrame, retention_df: pd.DataFrame):
    """Paired t-test across seeds between the top two models by overall robustness index.

    Per-seed robustness index = retention averaged across perturbation x
    severity for that seed, so the two models' seed-indexed samples are
    naturally paired (same seed = same perturbation draw for both models).
    """
    ranked = overall_df.sort_values("overall_robustness_index", ascending=False)
    if len(ranked) < 2:
        return None
    model_a, model_b = ranked["model"].iloc[0], ranked["model"].iloc[1]

    seed_level = (
        retention_df[retention_df["model"].isin([model_a, model_b])]
        .groupby(["model", "seed"])["retention"]
        .mean()
        .reset_index()
    )
    a = seed_level[seed_level["model"] == model_a].sort_values("seed")["retention"].to_numpy()
    b = seed_level[seed_level["model"] == model_b].sort_values("seed")["retention"].to_numpy()

    t_stat, p_value = stats.ttest_rel(a, b)
    return {"model_a": model_a, "model_b": model_b, "t_stat": t_stat, "p_value": p_value}


if __name__ == "__main__":
    overall_df, per_perturbation_df, retention_df = compute_robustness_index()

    config.ROBUSTNESS_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    overall_df.to_csv(config.ROBUSTNESS_INDEX_PATH, index=False)

    print(f"Robustness index (metric={config.ROBUSTNESS_METRIC}):\n")
    print(overall_df.to_string(index=False))

    print("\nPer-perturbation breakdown:\n")
    pivot = per_perturbation_df.pivot(index="model", columns="perturbation", values="robustness_index")
    print(pivot.to_string())

    over_one = overall_df[overall_df["overall_robustness_index"] > 1.0]
    if not over_one.empty:
        print(f"\nWARNING: robustness index > 1 for: {over_one['model'].tolist()} — check upstream computation.")

    ttest = paired_ttest_top_two(overall_df, retention_df)
    if ttest:
        print(
            f"\nPaired t-test ({config.ROBUSTNESS_METRIC} retention across seeds), "
            f"{ttest['model_a']} vs {ttest['model_b']}: "
            f"t={ttest['t_stat']:.4f}, p={ttest['p_value']:.4f}"
        )

    print(f"\nSaved to {config.ROBUSTNESS_INDEX_PATH}")
