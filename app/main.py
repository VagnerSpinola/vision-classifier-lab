from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from app.api.v1.batch_routes import router as batch_router
from app.api.v1.health_routes import router as health_router
from app.api.v1.predict_routes import router as predict_router
from app.core.settings import settings
from app.core.logging import get_logger, setup_logging
from app.inference.predictor import prediction_service
from app.monitoring.metrics import begin_request_timer, observe_request, update_model_status


setup_logging()
LOGGER = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    LOGGER.info("starting %s in %s mode", settings.project_name, settings.environment)
    try:
        prediction_service.load()
        metadata = prediction_service.metadata()
        update_model_status(
            loaded=metadata.loaded,
            model_version=metadata.model_version,
            backend=metadata.backend,
        )
    except FileNotFoundError:
        LOGGER.warning("no model checkpoint found at startup; /predict will return 503 until a checkpoint is available")
        update_model_status(False, settings.model_version, settings.inference_backend)
    yield


app = FastAPI(
    title="Vision MLOps Classifier API",
    version="1.0.0",
    description="Production-oriented image classification platform with PyTorch, ONNX, and MLOps instrumentation.",
    lifespan=lifespan,
)


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    started_at = begin_request_timer()
    try:
        response = await call_next(request)
    except Exception:
        observe_request(request.method, request.url.path, 500, started_at)
        raise
    observe_request(request.method, request.url.path, response.status_code, started_at)
    return response


app.include_router(health_router)
app.include_router(predict_router)
app.include_router(batch_router)