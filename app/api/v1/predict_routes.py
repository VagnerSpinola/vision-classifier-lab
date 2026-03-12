from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.v1._uploads import validate_image_upload
from app.core.logging import get_logger
from app.inference.predictor import prediction_service
from app.monitoring.metrics import record_error, record_prediction
from app.schemas.prediction import PredictionResponse, TopKPrediction


LOGGER = get_logger(__name__)
router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictionResponse:
    validate_image_upload(file)

    try:
        payload = await file.read()
        result = prediction_service.predict_bytes(payload)
        record_prediction(prediction_service.metadata().backend)
    except FileNotFoundError as error:
        LOGGER.exception("prediction failed because checkpoint is missing")
        record_error("/predict")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model checkpoint not available. Train or mount a checkpoint before calling /predict.",
        ) from error
    except Exception as error:
        LOGGER.exception("prediction request failed")
        record_error("/predict")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed.",
        ) from error

    return PredictionResponse(
        predicted_class=result.predicted_class,
        confidence=result.confidence,
        top_k=[TopKPrediction(class_name=item.class_name, score=item.score) for item in result.top_k],
    )