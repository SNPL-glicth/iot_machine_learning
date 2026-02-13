"""Explanation — value object raíz de la capa de explicabilidad.

Agrega todos los componentes del razonamiento cognitivo en un
único objeto inmutable y serializable:

    Explanation
    ├── signal_snapshot     — perfil de la señal de entrada
    ├── filter_snapshot     — resultado del filtrado
    ├── contribution_breakdown — desglose por engine
    ├── reasoning_trace     — fases del pipeline cognitivo
    ├── outcome             — resultado final (predicción/anomalía)
    └── audit_trace_id      — enlace al audit trail

Domain-pure.  Sin dependencias de infraestructura.
Extensible vía ``extra`` dict en cada componente.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .contribution_breakdown import ContributionBreakdown
from .reasoning_trace import ReasoningTrace
from .signal_snapshot import FilterSnapshot, SignalSnapshot


@dataclass(frozen=True)
class Outcome:
    """Resultado final de la inferencia.

    Captura qué decidió el motor (predicción, anomalía, o ambos)
    sin acoplar a las entidades de resultado específicas.

    Attributes:
        kind: Tipo de resultado (``"prediction"``, ``"anomaly"``,
            ``"prediction+anomaly"``).
        predicted_value: Valor predicho (si aplica).
        confidence: Confianza del resultado [0, 1].
        trend: Tendencia detectada.
        is_anomaly: True si se detectó anomalía.
        anomaly_score: Score de anomalía [0, 1] (si aplica).
        extra: Campos adicionales.
    """

    kind: str = "prediction"
    predicted_value: Optional[float] = None
    confidence: float = 0.0
    trend: str = "stable"
    is_anomaly: bool = False
    anomaly_score: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d: dict = {
            "kind": self.kind,
            "confidence": round(self.confidence, 4),
            "trend": self.trend,
            "is_anomaly": self.is_anomaly,
        }
        if self.predicted_value is not None:
            d["predicted_value"] = round(self.predicted_value, 6)
        if self.anomaly_score is not None:
            d["anomaly_score"] = round(self.anomaly_score, 6)
        if self.extra:
            d["extra"] = dict(self.extra)
        return d


@dataclass(frozen=True)
class Explanation:
    """Value object raíz de explicabilidad cognitiva.

    Compone todos los aspectos del razonamiento en un único
    objeto serializable.  Diseñado para ser adjuntado a cada
    ``Prediction`` o ``AnomalyResult`` como metadata estructurada.

    Attributes:
        series_id: Serie sobre la que se razonó.
        signal: Snapshot de la señal de entrada.
        filter: Snapshot del filtrado aplicado.
        contributions: Desglose de contribución por engine.
        trace: Traza de razonamiento (fases del pipeline).
        outcome: Resultado final de la inferencia.
        audit_trace_id: ID de enlace al audit trail.
        version: Versión del esquema de explicación.
    """

    series_id: str
    signal: SignalSnapshot = field(default_factory=SignalSnapshot.empty)
    filter: FilterSnapshot = field(default_factory=FilterSnapshot.empty)
    contributions: ContributionBreakdown = field(
        default_factory=ContributionBreakdown.empty
    )
    trace: ReasoningTrace = field(default_factory=ReasoningTrace.empty)
    outcome: Outcome = field(default_factory=Outcome)
    audit_trace_id: Optional[str] = None
    version: str = "1.0"

    @property
    def has_filter_data(self) -> bool:
        """True si se registró información de filtrado."""
        return self.filter.filter_name != "none"

    @property
    def has_trace(self) -> bool:
        """True si hay fases de razonamiento registradas."""
        return len(self.trace.phases) > 0

    @property
    def has_contributions(self) -> bool:
        """True si hay desglose de contribuciones."""
        return self.contributions.n_engines > 0

    @property
    def n_phases(self) -> int:
        """Número de fases de razonamiento."""
        return len(self.trace.phases)

    def to_dict(self) -> dict:
        d: dict = {
            "version": self.version,
            "series_id": self.series_id,
            "signal": self.signal.to_dict(),
            "outcome": self.outcome.to_dict(),
        }
        if self.has_filter_data:
            d["filter"] = self.filter.to_dict()
        if self.has_contributions:
            d["contributions"] = self.contributions.to_dict()
        if self.has_trace:
            d["trace"] = self.trace.to_dict()
        if self.audit_trace_id is not None:
            d["audit_trace_id"] = self.audit_trace_id
        return d

    @classmethod
    def minimal(cls, series_id: str) -> Explanation:
        """Factory para explicación mínima (solo series_id)."""
        return cls(series_id=series_id)
