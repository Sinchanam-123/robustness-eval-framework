"""
Phase 7, Day 16 — sanity tests for aggregation and the robustness index.

These read results.csv / baseline_metrics.csv, so they're skipped if the
offline pipeline (`python main.py`) hasn't been run yet — the perturbation
tests already cover the pure functions with no data dependency.
"""

import pytest

import config
from evaluation.aggregate import aggregate_results, load_results
from evaluation.robustness_index import compute_retention, compute_robustness_index

pytestmark = pytest.mark.skipif(
    not (config.RESULTS_CSV_PATH.exists() and config.BASELINE_METRICS_PATH.exists()),
    reason="results.csv / baseline_metrics.csv not found — run `python main.py` first",
)

# A robustness index of exactly 1 means no degradation; a hair above 1 is
# possible when a perturbation coincidentally helps a model. Allow a tiny
# tolerance so those legitimate cases don't fail the < 1 sanity check, while
# still catching a genuine upstream bug (e.g. index of 1.5).
INDEX_TOLERANCE = 0.02


def test_aggregated_row_count():
    # 3 models x 4 perturbations x 5 severities = 60 aggregated rows.
    agg = aggregate_results()
    expected = len(config.MODEL_NAMES) * len(config.PERTURBATION_NAMES) * len(
        config.SEVERITIES
    )
    assert len(agg) == expected


def test_overall_index_never_exceeds_one():
    overall_df, _, _ = compute_robustness_index()
    assert (overall_df["overall_robustness_index"] <= 1.0 + INDEX_TOLERANCE).all()


def test_overall_index_is_positive():
    overall_df, _, _ = compute_robustness_index()
    assert (overall_df["overall_robustness_index"] > 0.0).all()


def test_index_computed_for_every_model():
    overall_df, _, _ = compute_robustness_index()
    assert set(overall_df["model"]) == set(config.MODEL_NAMES)


def test_retention_matches_metric_over_baseline():
    import pandas as pd

    metric = config.ROBUSTNESS_METRIC
    results_df = load_results()
    baseline_df = pd.read_csv(config.BASELINE_METRICS_PATH)
    retention_df = compute_retention(results_df, baseline_df, metric)

    row = retention_df.iloc[0]
    baseline = baseline_df.set_index("model").loc[row["model"], metric]
    assert row["retention"] == pytest.approx(row[metric] / baseline)
