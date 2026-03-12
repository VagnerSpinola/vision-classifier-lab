# Deployment Guide

## Overview

This repository is designed for local orchestration with Docker Compose and a clean migration path toward Kubernetes, ECS, or Azure Container Apps.

## Deployment Steps

1. Train a model and export a checkpoint into `models/checkpoints/`.
2. Optionally export ONNX into `models/exported/` for CPU-oriented inference.
3. Copy `.env.example` to `.env` and configure runtime values.
4. Start the stack with `docker compose up --build`.
5. Verify the API on `/health` and metrics on `/metrics`.
6. Open Grafana on port `3000` and connect to Prometheus on `http://prometheus:9090`.

## Production Considerations

- Replace bind-mounted model artifacts with object storage or a model registry.
- Move MLflow metadata from local SQLite to PostgreSQL or MySQL.
- Route API logs and Prometheus metrics into centralized observability tooling.
- Add secrets management for environment variables and registry credentials.
- Gate deployment on CI test and Docker build success.