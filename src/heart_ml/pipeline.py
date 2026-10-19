"""Leakage-safe preprocessing and model construction."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart_ml.data import FEATURES

CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
ENGINEERED_NUMERIC_FEATURES = ["heart_rate_reserve", "max_hr_fraction"]
RAW_NUMERIC_FEATURES = [feature for feature in FEATURES if feature not in CATEGORICAL_FEATURES]
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + ENGINEERED_NUMERIC_FEATURES


class HeartFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add age-adjusted heart-rate features using the 220-minus-age heuristic."""

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "HeartFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("HeartFeatureEngineer expects a pandas DataFrame.")
        missing = sorted(set(FEATURES) - set(X.columns))
        if missing:
            raise ValueError(f"Input data is missing required features: {missing}")

        transformed = X.copy()
        estimated_max_hr = 220 - transformed["age"]
        transformed["heart_rate_reserve"] = estimated_max_hr - transformed["thalach"]
        transformed["max_hr_fraction"] = transformed["thalach"] / estimated_max_hr.where(
            estimated_max_hr > 0
        )
        return transformed


def make_preprocessor() -> ColumnTransformer:
    """Build imputers/encoders/scaling as part of the estimators' fitted pipeline."""
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_model(model_name: str, **parameters: object) -> Pipeline:
    """Return an end-to-end feature-engineering, preprocessing, and classifier pipeline."""
    classifiers = {
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight="balanced"
        ),
    }
    if model_name not in classifiers:
        raise ValueError(f"Unsupported model {model_name!r}; choose from {sorted(classifiers)}")
    classifier = classifiers[model_name].set_params(**parameters)
    return Pipeline(
        steps=[
            ("feature_engineering", HeartFeatureEngineer()),
            ("preprocessor", make_preprocessor()),
            ("classifier", classifier),
        ]
    )
