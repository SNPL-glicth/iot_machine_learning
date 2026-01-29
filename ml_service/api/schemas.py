"""Pydantic schemas for ML service API.

Request and response models for API endpoints.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""
    sensor_id: int = Field(..., gt=0, description="ID del sensor")
    horizon_minutes: int = Field(10, gt=0, le=1440, description="Horizonte de predicción en minutos")
    window: int = Field(60, gt=0, le=1000, description="Ventana de datos históricos")
    dedupe_minutes: int = Field(10, gt=0, le=1440, description="Minutos para deduplicación de eventos")


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint."""
    sensor_id: int
    model_id: int
    prediction_id: int
    predicted_value: float
    confidence: float
    target_timestamp: datetime
    horizon_minutes: int
    window: int


class HealthResponse(BaseModel):
    """Response schema for health endpoint."""
    status: str
    broker: dict | None = None
    version: str = "0.1.0"


class BrokerHealthResponse(BaseModel):
    """Response schema for broker health endpoint."""
    connected: bool
    type: str
    consumer_running: bool | None = None
    last_error: str | None = None
