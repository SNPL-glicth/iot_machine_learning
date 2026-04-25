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

import logging
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional

logger = logging.getLogger(__name__)

# IMP-5: allowed Outcome.kind values. Widen deliberately when a new
# kind is introduced so tests/lint catch unintended strings.
_ALLOWED_OUTCOME_KINDS: frozenset = frozenset({
    "prediction",
    "anomaly",
    "prediction+anomaly",
    "text_analysis",
    "analysis",
})
_ALLOWED_TRENDS: frozenset = frozenset({"up", "down", "stable"})

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
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # IMP-5: constructor invariants.
        if self.kind not in _ALLOWED_OUTCOME_KINDS:
            raise ValueError(
                f"Outcome.kind must be one of {sorted(_ALLOWED_OUTCOME_KINDS)}; got {self.kind!r}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"Outcome.confidence must be in [0.0, 1.0]; got {self.confidence!r}"
            )
        if self.trend not in _ALLOWED_TRENDS:
            raise ValueError(
                f"Outcome.trend must be one of {sorted(_ALLOWED_TRENDS)}; got {self.trend!r}"
            )
        if self.anomaly_score is not None and not (0.0 <= self.anomaly_score <= 1.0):
            raise ValueError(
                f"Outcome.anomaly_score must be in [0.0, 1.0] or None; got {self.anomaly_score!r}"
            )
        if self.is_anomaly and self.anomaly_score is None:
            logger.debug(
                "outcome_is_anomaly_without_score",
                extra={"kind": self.kind, "confidence": self.confidence},
            )
        # IMP-5: deep-freeze extra via MappingProxyType (read-only view over
        # a defensive copy). Mutating the original dict after construction
        # no longer leaks into the Outcome.
        object.__setattr__(self, "extra", MappingProxyType(dict(self.extra)))

    def with_extra(self, **kwargs: Any) -> "Outcome":
        """Return a new Outcome with ``extra`` merged with ``kwargs``.

        Kwargs override existing keys. The original Outcome is untouched.
        """
        merged = {**self.extra, **kwargs}
        return replace(self, extra=merged)

    def with_updates(self, **fields: Any) -> "Outcome":
        """Return a new Outcome with the supplied fields replaced.

        Thin wrapper around :func:`dataclasses.replace` kept for API
        symmetry with :meth:`with_extra`. All constructor invariants
        re-run on the new instance.
        """
        return replace(self, **fields)

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
