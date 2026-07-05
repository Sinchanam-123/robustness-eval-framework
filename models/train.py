"""
Phase 2, Day 3 — baseline models trained on clean data.

Trains Logistic Regression, Random Forest, and Gradient Boosting on the
frozen clean training split, evaluates each on the clean test split, and
saves both the fitted models (for reuse by the Phase 3 sweep, so it doesn't
retrain from scratch every run) and a one-row-per-model baseline metrics
table. These clean-test metrics are the severity=0 anchor that every later
robustness/retention number is computed relative to.

Run standalone: python -m models.train
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from config import (
    BASELINE_METRICS_PATH,
    MODEL_RANDOM_STATE,
    MODELS_DIR,
    TARGET_COLUMN,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
)


def load_train_test():
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)
    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]
    return X_train, y_train, X_test, y_test


def build_models():
    """One instance per name in config.MODEL_NAMES — keep the two in sync."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, random_state=MODEL_RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(random_state=MODEL_RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(
            random_state=MODEL_RANDOM_STATE
        ),
    }


def compute_metrics(y_true, y_pred) -> dict:
    # zero_division=0: at high perturbation severities a model can collapse
    # to predicting a single class, which would otherwise raise/warn on an
    # undefined precision or recall — 0.0 is the correct score in that case,
    # not an error.
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }


def train_and_evaluate():
    X_train, y_train, X_test, y_test = load_train_test()
    models = build_models()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = compute_metrics(y_test, preds)
        rows.append({"model": model_name, **metrics})

        joblib.dump(model, MODELS_DIR / f"{model_name}.joblib")

    metrics_df = pd.DataFrame(rows)
    BASELINE_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(BASELINE_METRICS_PATH, index=False)
    return metrics_df


if __name__ == "__main__":
    metrics_df = train_and_evaluate()
    print(metrics_df.to_string(index=False))
    print(f"\nModels saved to {MODELS_DIR}")
    print(f"Baseline metrics saved to {BASELINE_METRICS_PATH}")
