FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/heart_model.joblib
WORKDIR /app

COPY requirements-api.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements-api.txt
COPY src/ ./src/
COPY models/heart_model.joblib ./models/heart_model.joblib

EXPOSE 8000
USER 10001
CMD ["uvicorn", "heart_ml.api:app", "--host", "0.0.0.0", "--port", "8000"]
