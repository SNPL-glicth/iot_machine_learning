"""Data Transfer Objects para comunicación entre capas.

Los DTOs son objetos planos que cruzan fronteras de capas.
No contienen lógica de negocio — solo datos serializables.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple


@dataclass(frozen=True)
class PredictionDTO:
    """DTO para transferir predicciones entre capas.

    Attributes:
        sensor_id: ID del sensor.
        predicted_value: Valor predicho.
        confidence_score: Confianza numérica (0–1).
        confidence_level: Nivel cualitativo.
        trend: Dirección de tendencia.
        engine_name: Motor que generó la predicción.
        confidence_interval: Intervalo ``(lower, upper)`` si disponible.
        feature_contributions: Contribución de cada feature.
        explanation_text: Explicación legible.
        audit_trace_id: ID de trazabilidad.
    """

    sensor_id: int
    predicted_value: float
    confidence_score: float
    confidence_level: str
    trend: Literal["up", "down", "stable"]
    engine_name: str
    confidence_interval: Optional[Tuple[float, float]] = None
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    explanation_text: str = ""
    audit_trace_id: Optional[str] = None

    def to_dict(self) -> Dict[str, object]:
        """Serializa a dict para respuestas API."""
        result: Dict[str, object] = {
            "sensor_id": self.sensor_id,
            "predicted_value": self.predicted_value,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level,
            "trend": self.trend,
            "engine_name": self.engine_name,
        }
        if self.confidence_interval is not None:
            result["confidence_interval"] = {
                "lower": self.confidence_interval[0],
                "upper": self.confidence_interval[1],
            }
        if self.feature_contributions:
            result["feature_contributions"] = self.feature_contributions
        if self.explanation_text:
            result["explanation"] = self.explanation_text
        return result


@dataclass(frozen=True)
class AnomalyDTO:
    """DTO para transferir resultados de anomalía."""

    sensor_id: int
    is_anomaly: bool
    score: float
    severity: str
    method_votes: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    audit_trace_id: Optional[str] = None


@dataclass(frozen=True)
class PatternDTO:
    """DTO para transferir resultados de patrones."""

    sensor_id: int
    pattern_type: str
    confidence: float
    description: str = ""
    change_points: List[Dict[str, object]] = field(default_factory=list)
    spike_classification: Optional[str] = None
    current_regime: Optional[str] = None


@dataclass(frozen=True)
class SensorAnalysisDTO:
    """DTO completo de análisis de un sensor.

    Combina predicción + anomalía + patrón en un solo objeto
    para la capa de presentación.
    """

    sensor_id: int
    prediction: Optional[PredictionDTO] = None
    anomaly: Optional[AnomalyDTO] = None
    pattern: Optional[PatternDTO] = None
    processing_time_ms: float = 0.0
    audit_trace_id: Optional[str] = None
