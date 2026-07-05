"""
Phase 1, Day 2 — preprocessing pipeline for the Adult Census Income dataset.

Single entry point: preprocess_and_split() loads the raw data, cleans it,
one-hot encodes categoricals, scales numerics (fit on train only), performs
a stratified train/test split with a fixed seed, and freezes the result to
data/train.csv and data/test.csv.

Run standalone: python -m data.preprocessing
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    SPLIT_SEED,
    TRAIN_CSV_PATH,
    TEST_CSV_PATH,
)

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

NUMERIC_COLUMNS = [
    "age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
    "hours-per-week",
]

CATEGORICAL_COLUMNS = [
    c for c in COLUMN_NAMES if c not in NUMERIC_COLUMNS + [TARGET_COLUMN]
]


def load_raw() -> pd.DataFrame:
    return pd.read_csv(
        RAW_DATA_PATH,
        header=None,
        names=COLUMN_NAMES,
        skipinitialspace=True,
        na_values="?",
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # "?" (now NaN) categoricals are a legitimate missing-at-collection-time
    # category in this dataset, not something to drop rows over.
    for col in CATEGORICAL_COLUMNS:
        df[col] = df[col].fillna("Unknown")
    df[TARGET_COLUMN] = (df[TARGET_COLUMN].str.strip() == ">50K").astype(int)
    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(df, columns=CATEGORICAL_COLUMNS)


def preprocess_and_split():
    """Load -> clean -> encode -> scale -> stratified split -> freeze to disk.

    Returns (X_train, X_test, y_train, y_test).
    """
    df = clean(load_raw())
    df = encode(df)

    y = df[TARGET_COLUMN]
    X = df.drop(columns=[TARGET_COLUMN])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )

    # Scaler fit on train only — no leakage from test into the scaling stats.
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[NUMERIC_COLUMNS] = scaler.fit_transform(X_train[NUMERIC_COLUMNS])
    X_test[NUMERIC_COLUMNS] = scaler.transform(X_test[NUMERIC_COLUMNS])

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train.values
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test.values

    TRAIN_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    test_df.to_csv(TEST_CSV_PATH, index=False)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_and_split()
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"y_train class balance:\n{y_train.value_counts(normalize=True)}")
    print(f"Saved: {TRAIN_CSV_PATH}")
    print(f"Saved: {TEST_CSV_PATH}")
