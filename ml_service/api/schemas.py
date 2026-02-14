"""Pydantic schemas for ML service API.

Request and response models for API endpoints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request schema for prediction endpoint."""
    sensor_id: int = Field(..., gt=0, description="ID del sensor")
    horizon_minutes: int = Field(10, gt=0, le=1440, description="Horizonte de predicción en minutos")
    window: int = Field(60, gt=0, le=1000, description="Ventana de datos históricos")
    dedupe_minutes: int = Field(10, gt=0, le=1440, description="Minutos para deduplicación de eventos")


class StructuralAnalysisResponse(BaseModel):
    """Structural analysis of the signal."""
    regime: str = Field(description="Régimen dinámico: stable, trending, volatile, noisy")
    slope: float = Field(description="Pendiente local (tasa de cambio)")
    curvature: float = Field(description="Curvatura local (aceleración)")
    noise_ratio: float = Field(description="Ratio ruido/señal (σ/|μ|)")
    stability: float = Field(description="Indicador de estabilidad (0=estable, 1=inestable)")
    trend_strength: float = Field(description="Fuerza de la tendencia")
    mean: float = Field(description="Media de la serie")
    std: float = Field(description="Desviación estándar")
    n_points: int = Field(description="Puntos analizados")


class MetacognitiveResponse(BaseModel):
    """Metacognitive classification of the prediction."""
    certainty: str = Field(description="Nivel de certeza: high, moderate, low, very_low")
    disagreement: str = Field(description="Desacuerdo entre engines: none, consensus, mild, significant, severe")
    cognitive_stability: str = Field(description="Estabilidad cognitiva: stable, adapting, stressed, degraded")
    overfit_risk: str = Field(description="Riesgo de sobreajuste: not_applicable, high, moderate, low")
    engine_conflict: str = Field(description="Conflicto entre engines: none, aligned, mild_divergence, directional_conflict")


class PredictResponse(BaseModel):
    """Response schema for prediction endpoint.

    Campos base (backward compatible):
        sensor_id, model_id, prediction_id, predicted_value, confidence,
        target_timestamp, horizon_minutes, window

    Campos enriquecidos (nuevos, opcionales):
        trend, engine_used, confidence_level, structural_analysis,
        metacognitive, audit_trace_id, processing_time_ms
    """
    # --- Base fields (existing) ---
    sensor_id: int
    model_id: int
    prediction_id: int
    predicted_value: float
    confidence: float
    target_timestamp: datetime
    horizon_minutes: int
    window: int
    # --- Enrichment fields (new, optional for backward compat) ---
    trend: Optional[str] = Field(None, description="Tendencia: up, down, stable")
    engine_used: Optional[str] = Field(None, description="Motor de predicción utilizado")
    confidence_level: Optional[str] = Field(None, description="Nivel cualitativo: very_low, low, medium, high, very_high")
    structural_analysis: Optional[StructuralAnalysisResponse] = Field(None, description="Análisis estructural de la señal")
    metacognitive: Optional[MetacognitiveResponse] = Field(None, description="Clasificación metacognitiva")
    audit_trace_id: Optional[str] = Field(None, description="ID de trazabilidad ISO 27001")
    processing_time_ms: Optional[float] = Field(None, description="Tiempo de procesamiento en ms")


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
