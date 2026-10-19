import numpy as np
import pandas as pd
import pytest

from heart_ml.data import FEATURES
from heart_ml.pipeline import HeartFeatureEngineer, make_model


def sample_frame(rows=40):
    rng = np.random.default_rng(42)
    data = pd.DataFrame(rng.normal(size=(rows, len(FEATURES))), columns=FEATURES)
    for column, maximum in {
        "sex": 1,
        "cp": 4,
        "fbs": 1,
        "restecg": 2,
        "exang": 1,
        "slope": 3,
        "ca": 4,
        "thal": 7,
    }.items():
        data[column] = rng.integers(0, maximum + 1, size=rows)
    data.loc[0, "chol"] = np.nan
    return data


@pytest.mark.parametrize("model_name", ["logistic_regression", "random_forest"])
def test_pipeline_fits_with_missing_numeric_value(model_name):
    features = sample_frame()
    labels = pd.Series([0, 1] * 20)
    model = make_model(model_name).fit(features, labels)

    assert model.predict(features).shape == (40,)
    assert model.predict_proba(features).shape == (40, 2)


def test_feature_engineer_adds_age_adjusted_heart_rate_features():
    features = sample_frame(rows=1)
    features.loc[0, ["age", "thalach"]] = [60, 150]

    transformed = HeartFeatureEngineer().fit_transform(features)

    assert transformed.loc[0, "heart_rate_reserve"] == 10
    assert transformed.loc[0, "max_hr_fraction"] == pytest.approx(150 / 160)


def test_feature_engineer_leaves_undefined_fraction_for_imputation():
    features = sample_frame(rows=1)
    features.loc[0, ["age", "thalach"]] = [220, 150]

    transformed = HeartFeatureEngineer().fit_transform(features)

    assert np.isnan(transformed.loc[0, "max_hr_fraction"])


def test_unknown_model_fails_clearly():
    with pytest.raises(ValueError, match="Unsupported model"):
        make_model("svm")
