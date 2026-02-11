"""Value Objects para predicciones.

Representan el resultado de un motor de predicción, desacoplado de
cualquier implementación concreta (Taylor, baseline, ensemble, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Literal, Optional, Tuple


class PredictionConfidence(Enum):
    """Niveles cualitativos de confianza para uso en UI/alertas."""

    VERY_LOW = "very_low"    # < 0.2
    LOW = "low"              # 0.2 – 0.4
    MEDIUM = "medium"        # 0.4 – 0.7
    HIGH = "high"            # 0.7 – 0.9
    VERY_HIGH = "very_high"  # > 0.9

    @classmethod
    def from_score(cls, score: float) -> "PredictionConfidence":
        """Convierte score numérico a nivel cualitativo."""
        if score < 0.2:
            return cls.VERY_LOW
        if score < 0.4:
            return cls.LOW
        if score < 0.7:
            return cls.MEDIUM
        if score < 0.9:
            return cls.HIGH
        return cls.VERY_HIGH


@dataclass(frozen=True)
class Prediction:
    """Resultado de predicción del dominio.

    Desacoplado de la implementación del motor.  Contiene toda la
    información necesaria para persistencia, explicabilidad y auditoría.

    Attributes:
        series_id: Identificador de la serie predicha.
        predicted_value: Valor predicho.
        confidence_score: Confianza numérica (0.0–1.0).
        trend: Dirección de la tendencia.
        engine_name: Nombre del motor que generó la predicción.
        horizon_steps: Pasos adelante predichos.
        confidence_interval: Intervalo de confianza ``(lower, upper)``
            si el motor lo soporta.
        feature_contributions: Contribución de cada feature a la
            predicción (para explicabilidad).
        metadata: Información adicional específica del motor.
        audit_trace_id: ID de trazabilidad para auditoría ISO 27001.
    """

    series_id: str
    predicted_value: float
    confidence_score: float
    trend: Literal["up", "down", "stable"]
    engine_name: str
    horizon_steps: int = 1
    confidence_interval: Optional[Tuple[float, float]] = None
    feature_contributions: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)
    audit_trace_id: Optional[str] = None

    @property
    def confidence_level(self) -> PredictionConfidence:
        """Nivel cualitativo de confianza."""
        return PredictionConfidence.from_score(self.confidence_score)

    @property
    def has_confidence_interval(self) -> bool:
        """``True`` si el motor proveyó intervalo de confianza."""
        return self.confidence_interval is not None

    def to_audit_dict(self) -> Dict[str, object]:
        """Serializa para audit log (ISO 27001 A.12.4.1)."""
        return {
            "series_id": self.series_id,
            "predicted_value": self.predicted_value,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "trend": self.trend,
            "engine_name": self.engine_name,
            "horizon_steps": self.horizon_steps,
            "confidence_interval": self.confidence_interval,
            "audit_trace_id": self.audit_trace_id,
        }
