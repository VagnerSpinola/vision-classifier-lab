from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.api.v1._uploads import validate_image_uploads
from app.core.logging import get_logger
from app.inference.predictor import prediction_service
from app.monitoring.metrics import record_error, record_prediction
from app.schemas.prediction import BatchPredictionItem, BatchPredictionResponse, TopKPrediction


LOGGER = get_logger(__name__)
router = APIRouter(tags=["prediction"])


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(files: list[UploadFile] = File(...)) -> BatchPredictionResponse:
    validate_image_uploads(files)

    try:
        payloads = [await file.read() for file in files]
        results = prediction_service.predict_batch(payloads)
        record_prediction(prediction_service.metadata().backend, count=len(results))
    except FileNotFoundError as error:
        LOGGER.exception("batch prediction failed because the model artifact is missing")
        record_error("/predict/batch")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model artifact not available. Train or mount a model before calling /predict/batch.",
        ) from error
    except Exception as error:
        LOGGER.exception("batch prediction request failed")
        record_error("/predict/batch")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch prediction failed.",
        ) from error

    predictions = [
        BatchPredictionItem(
            filename=file.filename or f"image_{index}",
            predicted_class=result.predicted_class,
            confidence=result.confidence,
            top_k=[TopKPrediction(class_name=item.class_name, score=item.score) for item in result.top_k],
        )
        for index, (file, result) in enumerate(zip(files, results), start=1)
    ]
    return BatchPredictionResponse(predictions=predictions)