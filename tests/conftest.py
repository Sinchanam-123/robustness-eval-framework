"""
Shared pytest fixtures.

Adds the project root to sys.path so `config`, `data.*`, `stress_tests.*`,
and `evaluation.*` import cleanly when running `pytest tests/` from the root.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from data.preprocessing import NUMERIC_COLUMNS


@pytest.fixture
def synthetic_X():
    """A small preprocessed-style feature matrix.

    Mirrors the real X: the 6 scaled numeric columns plus a couple of one-hot
    categorical dummy columns. Self-contained so the perturbation tests don't
    require the offline pipeline (train.csv / results.csv) to have been run.
    The categorical dummies are here specifically to assert perturbations
    leave them untouched.
    """
    rng = np.random.default_rng(123)
    n = 200
    data = {col: rng.normal(0.0, 1.0, n) for col in NUMERIC_COLUMNS}
    data["workclass_Private"] = rng.integers(0, 2, n)
    data["sex_Male"] = rng.integers(0, 2, n)
    return pd.DataFrame(data)


@pytest.fixture
def categorical_columns():
    return ["workclass_Private", "sex_Male"]
