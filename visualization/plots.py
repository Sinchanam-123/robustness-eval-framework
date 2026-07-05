"""
Phase 5, Day 9 — visualization.

Primary chart: degradation curve (x=severity, y=metric, one line per model,
one chart per perturbation type) — this is the chart that actually
communicates the finding; a before/after bar chart hides the shape of the
decline. Each curve is anchored at severity=0 with the clean baseline metric
from Phase 2, so the plot shows the full decay from the unperturbed anchor.

Secondary: a heatmap of robustness index (model x perturbation type).

Run standalone: python -m visualization.plots
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config
from evaluation.aggregate import aggregate_results
from evaluation.robustness_index import compute_robustness_index


def plot_degradation_curves(aggregated_df: pd.DataFrame = None, metric: str = None, output_dir: Path = None):
    metric = metric or config.ROBUSTNESS_METRIC
    if aggregated_df is None:
        aggregated_df = aggregate_results()
    output_dir = output_dir or config.PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = pd.read_csv(config.BASELINE_METRICS_PATH).set_index("model")[metric]

    for perturbation in config.PERTURBATION_NAMES:
        fig, ax = plt.subplots(figsize=(7, 5))
        subset = aggregated_df[aggregated_df["perturbation"] == perturbation]

        for model in config.MODEL_NAMES:
            model_data = subset[subset["model"] == model].sort_values("severity")
            severities = [0.0] + model_data["severity"].tolist()
            means = [baseline[model]] + model_data[f"{metric}_mean"].tolist()
            stds = [0.0] + model_data[f"{metric}_std"].tolist()
            ax.errorbar(severities, means, yerr=stds, marker="o", capsize=3, label=model)

        ax.set_xlabel("Severity")
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"Degradation curve — {perturbation}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"degradation_{perturbation}.png", dpi=150)
        plt.close(fig)


def plot_robustness_heatmap(per_perturbation_df: pd.DataFrame = None, output_dir: Path = None):
    output_dir = output_dir or config.PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    if per_perturbation_df is None:
        _, per_perturbation_df, _ = compute_robustness_index()

    pivot = per_perturbation_df.pivot(index="model", columns="perturbation", values="robustness_index")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.9, vmax=1.0, ax=ax)
    ax.set_title("Robustness index — model x perturbation")
    fig.tight_layout()
    fig.savefig(output_dir / "robustness_heatmap.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_degradation_curves()
    plot_robustness_heatmap()
    print(f"Plots saved to {config.PLOTS_DIR}")
