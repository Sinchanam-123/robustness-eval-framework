"""
Phase 3, Day 4-5 — stress-test perturbation functions.

Every function signature is fn(X, severity, seed) (shift_distribution adds a
trailing `feature` arg, as scoped in the workplan) — never a bare fn(data).
Severities and seeds always come from config.py at the call site; nothing
here hardcodes its own values.

Scope note: X is the fully preprocessed feature matrix (6 scaled numeric
columns + one-hot encoded categorical columns). All four perturbations only
touch the numeric columns — adding Gaussian noise, missing values, or scaled
outliers to a one-hot indicator column would produce values that don't
correspond to any real category, so there's no coherent real-world reading
of "noisy" or "missing" categorical dummy. Noise/missingness/outliers/shift
are physically things that happen to continuous measurements, not to which
category a row belongs to.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import MISSING_VALUE_IMPUTE_STRATEGY, OUTLIER_SCALE, SHIFT_FEATURE
from data.preprocessing import NUMERIC_COLUMNS


def add_noise(X: pd.DataFrame, severity: float, seed: int) -> pd.DataFrame:
    """Add zero-mean Gaussian noise, scaled to each numeric column's own std."""
    rng = np.random.default_rng(seed)
    X_noisy = X.copy()
    numeric = X[NUMERIC_COLUMNS]
    std = numeric.std(axis=0) * severity
    noise = rng.normal(0, std, numeric.shape)
    X_noisy[NUMERIC_COLUMNS] = numeric.to_numpy() + noise
    return X_noisy


def introduce_missing(
    X: pd.DataFrame, severity: float, seed: int, strategy: str = None
) -> pd.DataFrame:
    """Null out numeric values at rate `severity`, then impute per `strategy`.

    strategy defaults to config.MISSING_VALUE_IMPUTE_STRATEGY ("median") —
    this imputation choice is part of the experiment design (it determines
    how much signal survives missingness), not an implementation detail.
    """
    strategy = strategy or MISSING_VALUE_IMPUTE_STRATEGY
    rng = np.random.default_rng(seed)
    X_missing = X.copy()
    numeric = X[NUMERIC_COLUMNS].copy()

    mask = rng.random(numeric.shape) < severity
    numeric[mask] = np.nan

    X_missing[NUMERIC_COLUMNS] = numeric.fillna(numeric.agg(strategy))
    return X_missing


def add_outliers(
    X: pd.DataFrame, severity: float, seed: int, scale: float = None
) -> pd.DataFrame:
    """Multiply a `severity`-fraction of rows' numeric values by `scale`."""
    scale = scale if scale is not None else OUTLIER_SCALE
    rng = np.random.default_rng(seed)
    X_out = X.copy()

    n_rows = int(len(X) * severity)
    if n_rows > 0:
        idx = rng.choice(len(X), n_rows, replace=False)
        X_out.iloc[idx, X_out.columns.get_indexer(NUMERIC_COLUMNS)] = (
            X_out.iloc[idx, X_out.columns.get_indexer(NUMERIC_COLUMNS)] * scale
        )
    return X_out


def shift_distribution(
    X: pd.DataFrame, severity: float, seed: int, feature: str = None
) -> pd.DataFrame:
    """Shift one feature's mean by severity * std — covariate shift only.

    This does not simulate label shift or concept shift: only the input
    feature's distribution moves, the true relationship between features
    and label is untouched. `seed` is accepted for a uniform fn(X, severity,
    seed) call signature across all perturbations, but is unused here — the
    shift itself is a deterministic mean translation, not a random draw.
    """
    feature = feature or SHIFT_FEATURE
    X_shift = X.copy()
    X_shift[feature] = X[feature] + X[feature].std() * severity
    return X_shift
