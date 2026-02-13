"""Value Objects para detección de anomalías.

Representan resultados de detección desacoplados de implementaciones
concretas (IsolationForest, LOF, Z-score, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


class AnomalySeverity(Enum):
    """Severidad de anomalía para decisiones de negocio."""

    NONE = "none"          # No es anomalía
    LOW = "low"            # Anomalía leve (monitorear)
    MEDIUM = "medium"      # Anomalía moderada (revisar)
    HIGH = "high"          # Anomalía severa (actuar)
    CRITICAL = "critical"  # Anomalía crítica (acción inmediata)

    @classmethod
    def from_score(
        cls,
        score: float,
        *,
        none_max: float = 0.3,
        low_max: float = 0.5,
        medium_max: float = 0.7,
        high_max: float = 0.9,
    ) -> "AnomalySeverity":
        """Convierte score numérico (0–1) a severidad.

        Args:
            score: Anomaly score in [0, 1].
            none_max: Scores below this are NONE.
            low_max: Scores below this are LOW.
            medium_max: Scores below this are MEDIUM.
            high_max: Scores below this are HIGH; above is CRITICAL.
        """
        if score < none_max:
            return cls.NONE
        if score < low_max:
            return cls.LOW
        if score < medium_max:
            return cls.MEDIUM
        if score < high_max:
            return cls.HIGH
        return cls.CRITICAL


@dataclass(frozen=True)
class AnomalyResult:
    """Resultado de detección de anomalía del dominio.

    Attributes:
        series_id: Identificador de la serie evaluada.
        is_anomaly: ``True`` si se detectó anomalía.
        score: Score de anomalía (0.0 = normal, 1.0 = anomalía fuerte).
        method_votes: Votos de cada método detector
            (e.g. ``{"isolation_forest": 0.8, "z_score": 0.6}``).
        confidence: Confianza en la detección (0.0–1.0).
        explanation: Explicación legible de la decisión.
        severity: Severidad derivada del score.
        context: Contexto adicional (régimen operacional, correlaciones).
        audit_trace_id: ID de trazabilidad ISO 27001.
    """

    series_id: str
    is_anomaly: bool
    score: float
    method_votes: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    explanation: str = ""
    severity: AnomalySeverity = AnomalySeverity.NONE
    context: Dict[str, object] = field(default_factory=dict)
    audit_trace_id: Optional[str] = None

    @classmethod
    def normal(cls, series_id: str) -> "AnomalyResult":
        """Factory para resultado normal (sin anomalía)."""
        return cls(
            series_id=series_id,
            is_anomaly=False,
            score=0.0,
            confidence=0.95,
            explanation="Valor dentro de rangos normales",
            severity=AnomalySeverity.NONE,
        )

    def to_audit_dict(self) -> Dict[str, object]:
        """Serializa para audit log (ISO 27001 A.12.4.1)."""
        return {
            "series_id": self.series_id,
            "is_anomaly": self.is_anomaly,
            "score": self.score,
            "method_votes": self.method_votes,
            "confidence": self.confidence,
            "severity": self.severity.value,
            "explanation": self.explanation,
            "audit_trace_id": self.audit_trace_id,
        }
