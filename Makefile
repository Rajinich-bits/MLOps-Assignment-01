.PHONY: install data eda train test lint api

install:
	python -m pip install -r requirements.txt
	python -m pip install -e ".[dev]"

data:
	python scripts/download_data.py

eda:
	python scripts/eda.py

train:
	python -m heart_ml.train --data data/raw/heart.csv --output models

test:
	pytest

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

api:
	uvicorn heart_ml.api:app --host 127.0.0.1 --port 8000
