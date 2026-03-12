from __future__ import annotations

from fastapi import APIRouter, Response

from app.core.settings import settings
from app.inference.predictor import prediction_service
from app.monitoring.metrics import render_metrics
from app.schemas.prediction import HealthResponse


router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    metadata = prediction_service.metadata()
    return HealthResponse(
        status="ok",
        model_loaded=metadata.loaded,
        model_version=metadata.model_version,
        backend=metadata.backend,
    )


@router.get(settings.metrics_path)
async def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)