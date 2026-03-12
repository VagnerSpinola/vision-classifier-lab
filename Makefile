PYTHON ?= python
CONFIG ?= experiments/configs/resnet18.yaml

.PHONY: install split validate train evaluate export run test lint mlflow docker-up docker-down

install:
	$(PYTHON) -m pip install -r requirements.txt

split:
	$(PYTHON) -m src.data.split_data --raw-dir data/raw --output-dir data/processed --clear-output

validate:
	$(PYTHON) -m src.data.validate_dataset --processed-dir data/processed

train:
	$(PYTHON) -m src.training.train --config $(CONFIG)

evaluate:
	$(PYTHON) -m src.evaluation.evaluate --config $(CONFIG)

export:
	$(PYTHON) -m src.export.export_onnx --config $(CONFIG)

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest -q

lint:
	ruff check .

mlflow:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./models

docker-up:
	docker compose up --build

docker-down:
	docker compose down