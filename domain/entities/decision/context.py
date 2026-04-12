"""DecisionContext — aggregated ML outputs for decision-making.

Domain-pure dataclass capturing all relevant information from
upstream ML pipeline without dependencies on specific ML types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..severity import SeverityResult
from .outcome import SimulatedOutcome


@dataclass(frozen=True)
class DecisionContext:
    """Aggregated ML outputs required for decision-making.

    Captures all relevant information from upstream ML pipeline
    without dependencies on specific ML types.

    Attributes:
        series_id: Identifier for the entity being analyzed
        severity: Severity classification from severity_rules
        confidence: Overall ML confidence [0, 1]
        is_anomaly: Whether an anomaly was detected
        anomaly_score: Anomaly score [0, 1] if applicable
        patterns: List of detected patterns with metadata
        predicted_value: Predicted future value if available
        trend: Detected trend direction
        monte_carlo_outcomes: Optional pre-computed scenario outcomes
        domain: Detected/assigned domain (infrastructure, security, etc.)
        audit_trace_id: Trace ID for audit logging
        extra: Additional ML outputs not captured above

        # Contextual enrichment fields (Paso 2)
        recent_anomaly_count: Anomalies in last 2 hours
        recent_anomaly_rate: Ratio anomalies/predictions in window [0, 1]
        consecutive_anomalies: Uninterrupted anomaly streak
        current_regime: Signal regime (STABLE, TRENDING, VOLATILE, NOISY)
        regime_duration_minutes: Time in current regime
        drift_score: Concept drift severity [0, 1]
        series_criticality: Business criticality (LOW, NORMAL, HIGH, CRITICAL)
        last_alert_timestamp: Unix timestamp of last alert emitted
        suppression_window_minutes: Cooldown period for duplicate alerts
    """

    series_id: str
    severity: SeverityResult = field(
        default_factory=lambda: SeverityResult(
            risk_level="NONE",
            severity="info",
            action_required=False,
            recommended_action="No action required",
        )
    )
    confidence: float = 0.0
    is_anomaly: bool = False
    anomaly_score: float = 0.0
    patterns: List[Dict[str, Any]] = field(default_factory=list)
    predicted_value: Optional[float] = None
    trend: str = "stable"
    monte_carlo_outcomes: Optional[List[SimulatedOutcome]] = None
    domain: str = ""
    audit_trace_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # Contexto enriquecido para decisiones contextuales (Paso 2)
    # Historial de anomalías de la serie (ventana 2 horas)
    recent_anomaly_count: int = 0
    recent_anomaly_rate: float = 0.0  # ratio anomalías / predicciones recientes
    consecutive_anomalies: int = 0  # sin interrupción hasta ahora

    # Régimen actual
    current_regime: str = "STABLE"
    regime_duration_minutes: float = 0.0  # cuánto lleva en este régimen
    drift_score: float = 0.0  # ya calculado en SignalProfile

    # Identidad y criticidad de la serie
    series_criticality: str = "NORMAL"  # LOW / NORMAL / HIGH / CRITICAL

    # Control de supresión
    last_alert_timestamp: Optional[float] = None
    suppression_window_minutes: float = 5.0

    @property
    def has_monte_carlo(self) -> bool:
        """True if Monte Carlo outcomes are available."""
        return self.monte_carlo_outcomes is not None and len(self.monte_carlo_outcomes) > 0

    @property
    def has_critical_pattern(self) -> bool:
        """True if any pattern has critical severity hint."""
        return any(
            p.get("severity_hint", "").lower() == "critical"
            for p in self.patterns
        )

    @property
    def action_required(self) -> bool:
        """True if upstream severity indicates action is required."""
        return self.severity.action_required

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "series_id": self.series_id,
            "severity": {
                "risk_level": self.severity.risk_level,
                "severity": self.severity.severity,
                "action_required": self.severity.action_required,
                "recommended_action": self.severity.recommended_action,
            },
            "confidence": round(self.confidence, 4),
            "is_anomaly": self.is_anomaly,
            "anomaly_score": round(self.anomaly_score, 4),
            "patterns": self.patterns,
            "predicted_value": self.predicted_value,
            "trend": self.trend,
            "monte_carlo_outcomes": (
                [o.to_dict() for o in self.monte_carlo_outcomes]
                if self.monte_carlo_outcomes else None
            ),
            "domain": self.domain,
            "audit_trace_id": self.audit_trace_id,
            # Contextual enrichment fields
            "recent_anomaly_count": self.recent_anomaly_count,
            "recent_anomaly_rate": round(self.recent_anomaly_rate, 4),
            "consecutive_anomalies": self.consecutive_anomalies,
            "current_regime": self.current_regime,
            "regime_duration_minutes": round(self.regime_duration_minutes, 2),
            "drift_score": round(self.drift_score, 4),
            "series_criticality": self.series_criticality,
            "last_alert_timestamp": self.last_alert_timestamp,
            "suppression_window_minutes": self.suppression_window_minutes,
        }
