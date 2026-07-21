"""
Phase 7, Day 16 — unit tests for the stress-test perturbation functions.

Core invariants verified for every perturbation:
- severity=0 returns the input unchanged (project non-negotiable),
- the same seed is reproducible,
- one-hot categorical columns are never touched,
- shape is preserved,
plus per-perturbation behavior and sanity at the extreme severity=1.
"""

import numpy as np
import pandas as pd
import pytest

import config
from data.preprocessing import NUMERIC_COLUMNS
from stress_tests.perturbations import (
    add_noise,
    add_outliers,
    introduce_missing,
    shift_distribution,
)

# Every perturbation as a uniform fn(X, severity, seed) callable.
ALL_PERTURBATIONS = [add_noise, introduce_missing, add_outliers, shift_distribution]
# Perturbations driven by the RNG seed (covariate_shift is deterministic).
SEEDED_PERTURBATIONS = [add_noise, introduce_missing, add_outliers]


@pytest.mark.parametrize("perturb_fn", ALL_PERTURBATIONS)
def test_severity_zero_returns_unchanged(synthetic_X, perturb_fn):
    result = perturb_fn(synthetic_X, 0.0, seed=0)
    pd.testing.assert_frame_equal(result, synthetic_X)


@pytest.mark.parametrize("perturb_fn", ALL_PERTURBATIONS)
def test_does_not_mutate_input(synthetic_X, perturb_fn):
    before = synthetic_X.copy()
    perturb_fn(synthetic_X, 0.4, seed=1)
    pd.testing.assert_frame_equal(synthetic_X, before)


@pytest.mark.parametrize("perturb_fn", ALL_PERTURBATIONS)
def test_shape_preserved(synthetic_X, perturb_fn):
    result = perturb_fn(synthetic_X, 0.5, seed=2)
    assert result.shape == synthetic_X.shape
    assert list(result.columns) == list(synthetic_X.columns)


@pytest.mark.parametrize("perturb_fn", ALL_PERTURBATIONS)
def test_categorical_columns_untouched(synthetic_X, perturb_fn, categorical_columns):
    result = perturb_fn(synthetic_X, 0.5, seed=3)
    for col in categorical_columns:
        pd.testing.assert_series_equal(result[col], synthetic_X[col])


@pytest.mark.parametrize("perturb_fn", SEEDED_PERTURBATIONS)
def test_same_seed_reproducible(synthetic_X, perturb_fn):
    a = perturb_fn(synthetic_X, 0.3, seed=7)
    b = perturb_fn(synthetic_X, 0.3, seed=7)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("perturb_fn", SEEDED_PERTURBATIONS)
def test_different_seed_differs(synthetic_X, perturb_fn):
    a = perturb_fn(synthetic_X, 0.3, seed=7)
    c = perturb_fn(synthetic_X, 0.3, seed=8)
    assert not a.equals(c)


# --- add_noise ---

def test_noise_changes_all_numeric_columns(synthetic_X):
    result = add_noise(synthetic_X, 0.5, seed=0)
    for col in NUMERIC_COLUMNS:
        assert not np.allclose(result[col], synthetic_X[col])


# --- introduce_missing ---

def test_missing_no_nan_after_imputation(synthetic_X):
    result = introduce_missing(synthetic_X, 0.5, seed=0)
    assert result.isna().sum().sum() == 0


def test_missing_extreme_severity_is_sane(synthetic_X):
    # severity=1 nulls every numeric value; the safety net must still return a
    # NaN-free frame (falls back to 0 when a column has no observed median).
    result = introduce_missing(synthetic_X, 1.0, seed=0)
    assert result.isna().sum().sum() == 0


# --- add_outliers ---

def test_outliers_scales_expected_row_fraction(synthetic_X):
    severity = 0.3
    result = add_outliers(synthetic_X, severity, seed=0)
    changed = (result[NUMERIC_COLUMNS] != synthetic_X[NUMERIC_COLUMNS]).any(axis=1)
    assert changed.sum() == int(len(synthetic_X) * severity)


def test_outliers_extreme_severity_scales_all_rows(synthetic_X):
    result = add_outliers(synthetic_X, 1.0, seed=0)
    changed = (result[NUMERIC_COLUMNS] != synthetic_X[NUMERIC_COLUMNS]).any(axis=1)
    assert changed.sum() == len(synthetic_X)


# --- shift_distribution ---

def test_shift_only_changes_target_feature(synthetic_X):
    result = shift_distribution(synthetic_X, 0.5, seed=0, feature=config.SHIFT_FEATURE)
    for col in synthetic_X.columns:
        if col == config.SHIFT_FEATURE:
            assert not np.allclose(result[col], synthetic_X[col])
        else:
            pd.testing.assert_series_equal(result[col], synthetic_X[col])


def test_shift_is_constant_mean_translation(synthetic_X):
    severity = 0.4
    feature = config.SHIFT_FEATURE
    result = shift_distribution(synthetic_X, severity, seed=0, feature=feature)
    delta = result[feature] - synthetic_X[feature]
    expected = synthetic_X[feature].std() * severity
    assert np.allclose(delta, expected)


def test_shift_is_seed_invariant(synthetic_X):
    # Covariate shift is a deterministic mean translation — seed must not matter.
    a = shift_distribution(synthetic_X, 0.3, seed=0)
    b = shift_distribution(synthetic_X, 0.3, seed=99)
    pd.testing.assert_frame_equal(a, b)
