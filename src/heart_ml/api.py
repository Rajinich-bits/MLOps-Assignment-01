"""FastAPI inference service with request metrics and privacy-conscious logs."""

import logging
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field
from starlette.responses import Response

from heart_ml.data import FEATURES

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("heart_ml.api")

REQUESTS = Counter(
    "heart_api_requests_total",
    "Number of API requests",
    ["method", "path", "status"],
)
LATENCY = Histogram(
    "heart_api_request_duration_seconds",
    "API request latency in seconds",
    ["method", "path"],
)


class HeartRecord(BaseModel):
    """One record matching all model input features."""

    model_config = ConfigDict(extra="forbid")

    age: Annotated[float, Field(ge=0, le=120)]
    sex: Literal[0, 1]
    cp: Literal[1, 2, 3, 4]
    trestbps: Annotated[float, Field(gt=0, le=300)]
    chol: Annotated[float, Field(ge=0, le=1000)]
    fbs: Literal[0, 1]
    restecg: Literal[0, 1, 2]
    thalach: Annotated[float, Field(gt=0, le=300)]
    exang: Literal[0, 1]
    oldpeak: Annotated[float, Field(ge=-10, le=20)]
    slope: Literal[1, 2, 3]
    ca: Literal[0, 1, 2, 3]
    thal: Literal[3, 6, 7]


class PredictionResponse(BaseModel):
    prediction: int
    confidence: float
    positive_probability: float


app = FastAPI(title="Heart Disease Educational Demo API", version="0.1.0")


@lru_cache(maxsize=1)
def get_model():
    model_path = Path(os.getenv("MODEL_PATH", "models/heart_model.joblib"))
    if not model_path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
    return joblib.load(model_path)


@app.middleware("http")
async def record_request_metrics(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        path = (
            request.url.path
            if request.url.path in {"/health", "/metrics", "/predict"}
            else "unmatched"
        )
        REQUESTS.labels(request.method, path, str(status)).inc()
        LATENCY.labels(request.method, path).observe(time.perf_counter() - start)
        logger.info(
            "request method=%s path=%s status=%s",
            request.method,
            path,
            status,
        )


@app.get("/health")
def health() -> dict[str, str]:
    model_path = Path(os.getenv("MODEL_PATH", "models/heart_model.joblib"))
    if not model_path.is_file():
        raise HTTPException(status_code=503, detail="Model artifact is not available")
    return {"status": "ok"}


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(record: HeartRecord) -> PredictionResponse:
    try:
        model = get_model()
    except (FileNotFoundError, OSError, ValueError) as error:
        logger.error("Model loading failed: %s", error)
        raise HTTPException(status_code=503, detail="Model artifact is unavailable") from error
    frame = pd.DataFrame([record.model_dump()], columns=FEATURES)
    predictions = model.predict(frame)
    probabilities = model.predict_proba(frame)[:, 1]
    positive_probability = float(probabilities[0])
    return PredictionResponse(
        prediction=int(predictions[0]),
        confidence=max(positive_probability, 1 - positive_probability),
        positive_probability=positive_probability,
    )
