"""
Phase 3, Day 6 — the sweep runner.

Runs every (model x perturbation x severity x seed) combination and writes
one row per run to results.csv. This is what turns a single noisy sample
into a statistically averaged result: 3 models x 4 perturbations x 5
severities x 5 seeds = 300 evaluation runs. Every later phase (aggregation,
robustness index, plots, insights) reads only from this file.

Track A scope: models are loaded already trained on clean data (Phase 2);
only the test set is perturbed here. Training on perturbed data (Track B)
is out of scope for this sweep.

Run standalone: python -m stress_tests.sweep
"""

import sys
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd

import config
from models.train import compute_metrics, load_train_test
from stress_tests.perturbations import (
    add_noise,
    add_outliers,
    introduce_missing,
    shift_distribution,
)


def load_models() -> dict:
    return {
        name: joblib.load(config.MODELS_DIR / f"{name}.joblib")
        for name in config.MODEL_NAMES
    }


def get_perturbations() -> dict:
    """One entry per name in config.PERTURBATION_NAMES — keep the two in sync.

    Every value is callable as fn(X, severity, seed); covariate_shift binds
    its `feature` argument to config.SHIFT_FEATURE via partial so the sweep
    loop can call all four perturbations uniformly.
    """
    return {
        "noise": add_noise,
        "missing_values": introduce_missing,
        "outliers": add_outliers,
        "covariate_shift": partial(shift_distribution, feature=config.SHIFT_FEATURE),
    }


def run_sweep() -> pd.DataFrame:
    _, _, X_test, y_test = load_train_test()
    models = load_models()
    perturbations = get_perturbations()

    rows = []
    for model_name, model in models.items():
        for perturbation_name, perturb_fn in perturbations.items():
            for severity in config.SEVERITIES:
                for seed in config.SEEDS:
                    X_test_perturbed = perturb_fn(X_test, severity, seed)
                    preds = model.predict(X_test_perturbed)
                    metrics = compute_metrics(y_test, preds)
                    rows.append(
                        {
                            "model": model_name,
                            "perturbation": perturbation_name,
                            "severity": severity,
                            "seed": seed,
                            **metrics,
                        }
                    )

    results_df = pd.DataFrame(rows)
    config.RESULTS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(config.RESULTS_CSV_PATH, index=False)
    return results_df


if __name__ == "__main__":
    results_df = run_sweep()
    print(f"{len(results_df)} rows written to {config.RESULTS_CSV_PATH}")
    print(results_df.head())
