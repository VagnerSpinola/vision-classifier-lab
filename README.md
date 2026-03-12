# vision-mlops-classifier

Production-grade image classification platform built with PyTorch, FastAPI, MLflow, ONNX, Prometheus, Grafana, Docker Compose, and GitHub Actions.

This repository is designed to look and feel like a serious open-source ML systems project rather than a notebook-first demo. It combines model training, experiment tracking, evaluation, export, serving, and observability in one cohesive codebase.

## Why This Repository Exists

Most image classification portfolio projects stop at one of these points:

- a single notebook
- a training script with no deployment path
- a toy API with no monitoring
- a model artifact with no evaluation or registry story

vision-mlops-classifier is intentionally broader. It demonstrates how a modern computer vision system is structured when the goal is not just to train a model, but to operate it.

## Highlights

- Transfer learning with `resnet18`, `efficientnet_b0`, and `mobilenet_v3_small`
- Config-driven training and evaluation workflows
- Class imbalance handling through weighted loss or weighted sampling
- MLflow experiment tracking for parameters, metrics, and artifacts
- Test-set evaluation with confusion matrix, classification report, and error analysis
- Grad-CAM utilities for explainability
- ONNX export for production-oriented inference
- FastAPI service with single and batch prediction endpoints
- Prometheus metrics and Grafana dashboard configuration
- Docker Compose stack for API, MLflow, Prometheus, and Grafana
- GitHub Actions workflows for CI and Docker build validation

## Core Capabilities

| Area | Included |
| --- | --- |
| Model training | PyTorch training pipeline with train, validation, and test phases |
| Architectures | ResNet18, EfficientNet-B0, MobileNetV3-Small |
| Experiment management | YAML configs, MLflow logging, reproducible runs |
| Evaluation | Accuracy, precision, recall, F1-score, confusion matrix, classification report |
| Error analysis | Misclassification export for difficult samples |
| Explainability | Grad-CAM support |
| Model packaging | Checkpointing, local registry path, ONNX export |
| Serving | FastAPI inference API with online and batch prediction |
| Monitoring | Prometheus metrics, Grafana dashboard, structured logs |
| Delivery | Docker, Docker Compose, GitHub Actions |

## Architecture Overview

```mermaid
flowchart TB
    classDef data fill:#E8F3FF,stroke:#1D4ED8,stroke-width:1px,color:#0F172A;
    classDef train fill:#ECFDF5,stroke:#059669,stroke-width:1px,color:#0F172A;
    classDef artifact fill:#FFF7ED,stroke:#EA580C,stroke-width:1px,color:#0F172A;
    classDef serve fill:#F5F3FF,stroke:#7C3AED,stroke-width:1px,color:#0F172A;
    classDef ops fill:#FEF2F2,stroke:#DC2626,stroke-width:1px,color:#0F172A;

    subgraph DataLayer[Data Layer]
        Raw[Raw Images<br/>data/raw]:::data
        Split[Split and Validate<br/>src/data]:::data
        Processed[Versioned Processed Splits<br/>train | val | test]:::data
        Raw --> Split --> Processed
    end

    subgraph TrainingLayer[Training and Experimentation]
        Config[Experiment Configs<br/>experiments/configs]:::train
        Trainer[PyTorch Training Pipeline<br/>augmentation | imbalance handling | callbacks]:::train
        Eval[Evaluation and Explainability<br/>metrics | confusion matrix | Grad-CAM]:::train
        MLflow[MLflow Tracking<br/>params | metrics | artifacts]:::ops
        Config --> Trainer
        Processed --> Trainer
        Trainer --> Eval
        Trainer --> MLflow
        Eval --> MLflow
    end

    subgraph ArtifactLayer[Model Artifacts]
        Checkpoint[Best Checkpoint<br/>models/checkpoints]:::artifact
        Registry[Registry Path<br/>models/registry]:::artifact
        Onnx[ONNX Export<br/>models/exported]:::artifact
        Trainer --> Checkpoint
        Checkpoint --> Registry
        Checkpoint --> Onnx
    end

    subgraph ServingLayer[Serving and Inference]
        API[FastAPI Service<br/>/predict | /predict/batch | /health | /metrics]:::serve
        TorchRuntime[PyTorch Runtime]:::serve
        OnnxRuntime[ONNX Runtime]:::serve
        Registry --> API
        Onnx --> API
        API --> TorchRuntime
        API --> OnnxRuntime
    end

    subgraph OperationsLayer[Operations and Observability]
        CI[GitHub Actions CI/CD<br/>lint | tests | docker build]:::ops
        Prom[Prometheus Metrics]:::ops
        Grafana[Grafana Dashboards]:::ops
        Drift[Drift Monitoring<br/>prediction distribution checks]:::ops
        API --> Prom --> Grafana
        API --> Drift
        CI --> API
    end
```

## Repository Layout

```text
vision-mlops-classifier/
├── app/                      # API, inference runtime, schemas, monitoring
├── src/                      # Data, training, evaluation, export, explainability
├── mlops/                    # MLflow, Prometheus, Grafana, drift monitoring
├── scripts/                  # CLI wrappers for common workflows
├── experiments/configs/      # Experiment definitions
├── data/                     # Raw, processed, external, and sample assets
├── models/                   # Checkpoints, exports, registry artifacts
├── notebooks/                # Exploration and inspection
├── tests/                    # API, training, preprocessing, inference tests
├── .github/workflows/        # CI and Docker workflows
└── infrastructure/           # Docker and deployment documentation
```

## Project Structure In Practice

The repository is organized around distinct concerns.

- `src/data` handles splitting, validation, transforms, and dataloaders.
- `src/training` contains the configurable training pipeline, callbacks, checkpointing, and loss setup.
- `src/evaluation` produces metrics and artifacts that are useful for both model selection and debugging.
- `src/export` handles ONNX export for deployment-oriented runtimes.
- `app` contains the online serving layer, runtime preprocessing, schemas, and monitoring hooks.
- `mlops` contains experiment tracking and observability assets that would normally sit alongside the application in a real platform repo.

## Quickstart

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Create your local environment file

```bash
copy .env.example .env
```

### 3. Prepare a dataset

Expected raw dataset layout:

```text
data/raw/
├── class_a/
│   ├── image_001.jpg
│   └── image_002.jpg
└── class_b/
    ├── image_101.jpg
    └── image_102.jpg
```

### 4. Split and validate the dataset

```bash
make split
make validate
```

### 5. Train a baseline model

```bash
make train CONFIG=experiments/configs/resnet18.yaml
```

### 6. Evaluate the trained checkpoint

```bash
make evaluate CONFIG=experiments/configs/resnet18.yaml
```

### 7. Start the API

```bash
make run
```

## Supported Experiment Presets

- `experiments/configs/resnet18.yaml`
- `experiments/configs/efficientnet_b0.yaml`
- `experiments/configs/mobilenet_v3.yaml`

Each config controls architecture, image size, optimizer settings, class count, augmentation, imbalance strategy, MLflow destination, and ONNX export path.

## Training Pipeline

The training system is built around configuration-driven experimentation instead of hard-coded scripts. A typical run performs the following steps:

1. Load experiment settings from YAML.
2. Build dataset splits and runtime transforms.
3. Configure imbalance handling with weighted loss or weighted sampling.
4. Initialize a pretrained torchvision backbone and replace the classifier head.
5. Train with train and validation phases.
6. Save the best checkpoint based on the configured validation metric.
7. Log metrics and artifacts to MLflow.
8. Evaluate on the test set and persist outputs for inspection.

Main implementation files:

- `src/training/train.py`
- `src/training/trainer.py`
- `src/training/callbacks.py`
- `src/training/checkpoint.py`
- `src/training/loss.py`

## Evaluation and Error Analysis

The evaluation workflow produces artifacts that are useful for more than just leaderboard-style reporting.

Outputs include:

- accuracy
- precision
- recall
- F1-score
- confusion matrix image
- classification report in JSON and CSV
- error analysis CSV for misclassified samples

This is implemented through:

- `src/evaluation/evaluate.py`
- `src/evaluation/metrics.py`
- `src/evaluation/confusion_matrix.py`
- `src/evaluation/classification_report.py`
- `src/evaluation/error_analysis.py`

## Explainability

Grad-CAM support is included in `src/explainability/gradcam.py`. The goal is to make the repository useful for model debugging and stakeholder review, not only raw prediction serving.

## Experiment Tracking With MLflow

MLflow integration is implemented in `mlops/mlflow/tracking.py`.

For each run, the project logs:

- flattened experiment parameters
- epoch-level training metrics
- validation metrics
- final evaluation metrics
- history artifacts
- confusion matrix and report artifacts

Start MLflow locally with:

```bash
make mlflow
```

Or run the full stack with Docker Compose.

## Model Registry Concept

The repository uses a simple local model registry pattern. The best checkpoint is written to `models/checkpoints/` and promoted into `models/registry/` as the currently selected artifact for serving.

In a larger production system, this same pattern would typically map to:

- MLflow Model Registry
- S3 or GCS plus metadata storage
- SageMaker Model Registry
- an internal artifact registry

## Serving Layer

The FastAPI service supports both PyTorch and ONNX inference backends, controlled through environment configuration.

Available endpoints:

- `GET /health`
- `GET /metrics`
- `POST /predict`
- `POST /predict/batch`

### Health Check

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_version": "epoch-10",
  "backend": "torch"
}
```

### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@data/samples/example.jpg"
```

Example response:

```json
{
  "predicted_class": "class_a",
  "confidence": 0.9734,
  "top_k": [
    {"class_name": "class_a", "score": 0.9734},
    {"class_name": "class_b", "score": 0.0266}
  ]
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "files=@data/samples/example_1.jpg" \
  -F "files=@data/samples/example_2.jpg"
```

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

## Observability

The repository includes three separate observability layers.

- Structured logs for training and inference execution
- Prometheus metrics for service behavior
- Grafana dashboard configuration for fast local inspection

Prometheus and Grafana assets are stored in:

- `mlops/monitoring/prometheus.yml`
- `mlops/monitoring/grafana_dashboard.json`

## Drift Monitoring

The project includes a lightweight drift-monitoring placeholder in `mlops/drift/drift_monitor.py`. It compares prediction distributions between reference and current windows and is intended as a foundation for more advanced statistical monitoring.

## ONNX Export and Deployment

Export a model with:

```bash
make export CONFIG=experiments/configs/resnet18.yaml
```

The exported model can be used by the ONNX runtime inference path for leaner CPU-focused deployments.

Deployment notes are documented in `infrastructure/deployment/deployment_guide.md`.

## Docker Usage

Start the full local stack with:

```bash
docker compose up --build
```

This launches:

- FastAPI API on port `8000`
- MLflow on port `5000`
- Prometheus on port `9090`
- Grafana on port `3000`

## CI/CD

GitHub Actions workflows are included for both repository validation and image build automation.

- `.github/workflows/ci.yml` installs dependencies, runs Ruff, validates imports, and runs pytest.
- `.github/workflows/docker.yml` builds the API container image.

This gives the repository a credible delivery story instead of leaving validation as a manual step.

## Developer Commands

```bash
make install
make split
make validate
make train
make evaluate
make export
make run
make test
make mlflow
make docker-up
make docker-down
```

## Notebook

The exploratory notebook in `notebooks/exploratory_analysis.ipynb` is intentionally supplementary. It is there for inspection and analysis, not as the primary interface for the project.

## Production Notes

- Python 3.11 is the target runtime for CI and container images.
- Checkpoints and exported artifacts are intentionally excluded from source control.
- Environment variables control serving behavior, backend selection, model paths, and observability settings.
- The project is structured so it can grow toward a hosted registry, queue-backed async inference, or Kubernetes deployment without major reorganization.

## Future Improvements

- asynchronous offline batch jobs backed by queues and object storage
- richer drift detection using embeddings or feature statistics
- distributed training and mixed precision profiles
- hosted registry integration and release approvals
- canary deployment support and rollback automation

## License

This project is released under the terms of the license in `LICENSE`.
