from __future__ import annotations

from pydantic import BaseModel, Field


class TopKPrediction(BaseModel):
    class_name: str = Field(..., examples=["cat"])
    score: float = Field(..., ge=0.0, le=1.0, examples=[0.91])


class PredictionResponse(BaseModel):
    predicted_class: str = Field(..., examples=["cat"])
    confidence: float = Field(..., ge=0.0, le=1.0, examples=[0.9821])
    top_k: list[TopKPrediction]


class BatchPredictionItem(PredictionResponse):
    filename: str = Field(..., examples=["sample.jpg"])


class BatchPredictionResponse(BaseModel):
    predictions: list[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok"])
    model_loaded: bool = Field(..., examples=[True])
    model_version: str = Field(..., examples=["epoch-10"])
    backend: str = Field(..., examples=["torch"])