import numpy as np
from fastapi.testclient import TestClient

import heart_ml.api as api


class FakeModel:
    def predict(self, frame):
        return np.zeros(len(frame), dtype=int)

    def predict_proba(self, frame):
        return np.tile([0.8, 0.2], (len(frame), 1))


def valid_record():
    return {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 1,
        "ca": 0,
        "thal": 3,
    }


def test_predict_response_and_metrics(monkeypatch):
    monkeypatch.setattr(api, "get_model", lambda: FakeModel())
    client = TestClient(api.app)

    response = client.post("/predict", json=valid_record())

    assert response.status_code == 200
    assert response.json() == {
        "prediction": 0,
        "confidence": 0.8,
        "positive_probability": 0.2,
    }
    assert client.get("/metrics").status_code == 200


def test_predict_rejects_missing_features():
    client = TestClient(api.app)

    response = client.post("/predict", json={"age": 63})

    assert response.status_code == 422


def test_predict_rejects_unsupported_categorical_values():
    record = valid_record()
    record["thal"] = 5

    response = TestClient(api.app).post("/predict", json=record)

    assert response.status_code == 422
