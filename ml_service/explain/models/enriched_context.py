"""Enriched context model for contextual explainer."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class EnrichedContext:
    """Contexto enriquecido para el explainer."""
    
    # Información del sensor
    sensor_id: int
    sensor_type: str
    sensor_name: str
    unit: str
    
    # Información del dispositivo
    device_id: int
    device_name: str
    device_type: str
    location: str
    
    # Valores y predicción
    current_value: float
    predicted_value: float
    trend: str
    confidence: float
    horizon_minutes: int
    
    # Anomalía
    is_anomaly: bool
    anomaly_score: float
    
    # Umbrales del usuario
    user_threshold_min: Optional[float]
    user_threshold_max: Optional[float]
    
    # Historial reciente
    recent_avg: Optional[float]
    recent_min: Optional[float]
    recent_max: Optional[float]
    recent_std: Optional[float]
    
    # Eventos correlacionados
    correlated_events: list[dict]
    
    # Historial de eventos similares
    similar_events_count: int
    last_similar_event_at: Optional[datetime]
    
    def to_dict(self) -> dict:
        return {
            "sensor": {
                "id": self.sensor_id,
                "type": self.sensor_type,
                "name": self.sensor_name,
                "unit": self.unit,
            },
            "device": {
                "id": self.device_id,
                "name": self.device_name,
                "type": self.device_type,
                "location": self.location,
            },
            "prediction": {
                "current_value": self.current_value,
                "predicted_value": self.predicted_value,
                "trend": self.trend,
                "confidence": self.confidence,
                "horizon_minutes": self.horizon_minutes,
            },
            "anomaly": {
                "is_anomaly": self.is_anomaly,
                "score": self.anomaly_score,
            },
            "thresholds": {
                "min": self.user_threshold_min,
                "max": self.user_threshold_max,
            },
            "recent_stats": {
                "avg": self.recent_avg,
                "min": self.recent_min,
                "max": self.recent_max,
                "std": self.recent_std,
            },
            "correlated_events": self.correlated_events,
            "history": {
                "similar_events_count": self.similar_events_count,
                "last_similar_event_at": self.last_similar_event_at.isoformat() if self.last_similar_event_at else None,
            },
        }
