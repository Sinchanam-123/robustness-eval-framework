"""
Offline pipeline entrypoint: preprocessing -> training -> sweep -> results.csv.

This is the single source of truth for the project. results.csv (and the
baseline metrics / saved models it depends on) should only ever be produced
by running this file — the FastAPI backend (Phase 6) only reads results.csv,
it never recomputes it.

Run: python main.py
"""

from data.preprocessing import preprocess_and_split
from models.train import train_and_evaluate
from stress_tests.sweep import run_sweep


def main():
    print("[1/3] Preprocessing + train/test split...")
    preprocess_and_split()

    print("[2/3] Training baseline models...")
    baseline_metrics = train_and_evaluate()
    print(baseline_metrics.to_string(index=False))

    print("[3/3] Running stress-test sweep...")
    results_df = run_sweep()
    print(f"{len(results_df)} rows written to results.csv")


if __name__ == "__main__":
    main()
