"""
Central configuration for the robustness evaluation framework.

Every severity, seed, model, and path constant used anywhere in the project
must be read from this file. No other module hardcodes its own values.
"""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "adult.data"
COLUMN_NAMES_PATH = PROJECT_ROOT / "data" / "raw" / "adult.names"
TRAIN_CSV_PATH = PROJECT_ROOT / "data" / "train.csv"
TEST_CSV_PATH = PROJECT_ROOT / "data" / "test.csv"
FEATURE_STATS_PATH = PROJECT_ROOT / "data" / "feature_stats.md"
MODELS_DIR = PROJECT_ROOT / "models" / "artifacts"
RESULTS_CSV_PATH = PROJECT_ROOT / "results.csv"
BASELINE_METRICS_PATH = PROJECT_ROOT / "models" / "baseline_metrics.csv"
AGGREGATED_RESULTS_PATH = PROJECT_ROOT / "evaluation" / "aggregated_results.csv"
ROBUSTNESS_INDEX_PATH = PROJECT_ROOT / "evaluation" / "robustness_index.csv"
PLOTS_DIR = PROJECT_ROOT / "visualization" / "plots"

# --- Dataset ---
TARGET_COLUMN = "income"
TEST_SIZE = 0.2
SPLIT_SEED = 42  # fixed seed for the frozen train/test split (independent of sweep SEEDS)

# --- Stress-test sweep parameters ---
# Every stress-test function signature is fn(X, severity, seed) — never a bare
# fn(data). These two lists are the only place severities/seeds are defined.
SEVERITIES = [0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = [0, 1, 2, 3, 4]

# --- Models ---
# Logistic Regression, Random Forest, Gradient Boosting only.
# Decision Tree is deliberately excluded — same family as Random Forest.
MODEL_NAMES = ["logistic_regression", "random_forest", "gradient_boosting"]
MODEL_RANDOM_STATE = 42  # fixed seed for model training reproducibility

# --- Perturbations ---
# shift_distribution simulates covariate shift only (shifts one feature's mean
# by severity * std). It does not simulate label shift or concept shift.
PERTURBATION_NAMES = ["noise", "missing_values", "outliers", "covariate_shift"]
MISSING_VALUE_IMPUTE_STRATEGY = "median"  # strategy used by introduce_missing
OUTLIER_SCALE = 5  # multiplier applied to perturbed rows in add_outliers
SHIFT_FEATURE = "age"  # feature whose mean is shifted by shift_distribution

# --- Robustness index ---
# retention(severity) = metric(severity) / metric(severity=0); robustness_index
# = mean(retention across tested severities), averaged across perturbation
# types. ROBUSTNESS_METRIC picks which per-run metric column that's computed on.
ROBUSTNESS_METRIC = "accuracy"
ROBUSTNESS_INDEX_LOW_THRESHOLD = 0.7  # below this, a model "degrades significantly"
