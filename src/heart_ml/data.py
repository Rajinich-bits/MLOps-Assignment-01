"""Loading and validating the processed Cleveland Heart Disease data."""

from pathlib import Path

import pandas as pd

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET = "target"
COLUMNS = [*FEATURES, TARGET]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a headerless UCI CSV or a named CSV and map disease levels to 0/1."""
    path = Path(path)
    named_columns = {str(column).strip().lower() for column in pd.read_csv(path, nrows=0).columns}
    is_named_csv = bool(named_columns.intersection(FEATURES + [TARGET, "num"]))
    if is_named_csv:
        normalized_target = TARGET if TARGET in named_columns else "num"
        missing = sorted(set(FEATURES + [normalized_target]) - named_columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")
        raw = pd.read_csv(path, na_values=["?"], skipinitialspace=True)
        raw.columns = [str(column).strip().lower() for column in raw.columns]
        if "num" in raw.columns and TARGET not in raw.columns:
            raw = raw.rename(columns={"num": TARGET})
        raw = raw.loc[:, COLUMNS]
    else:
        raw = pd.read_csv(path, header=None, na_values=["?"], skipinitialspace=True)
        if raw.shape[1] != len(COLUMNS):
            raise ValueError(f"Expected {len(COLUMNS)} columns for UCI data; found {raw.shape[1]}")
        raw.columns = COLUMNS

    for column in COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    raw = raw.dropna(subset=[TARGET]).copy()
    raw[TARGET] = (raw[TARGET] > 0).astype("int64")
    if raw.empty:
        raise ValueError(f"No rows with a valid target found in {path}")
    return raw.reset_index(drop=True)


def save_cleaned_dataset(source: str | Path, destination: str | Path) -> Path:
    """Save normalized features and binary target while preserving imputable missing values."""
    cleaned = load_dataset(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(destination, index=False)
    return destination
