"""Módulo de contexto para decisiones ML."""

from .decision_context import (
    ActionUrgency,
    ImpactLevel,
    RecommendedAction,
    ImpactAssessment,
    EscalationInfo,
    DecisionContext,
    DecisionContextBuilder,
)
from .operational_context import (
    WorkShift,
    StaffAvailability,
    ProductionImpact,
    OperationalContext,
    OperationalContextBuilder,
    adjust_severity_with_context,
)

__all__ = [
    # Decision context
    "ActionUrgency",
    "ImpactLevel",
    "RecommendedAction",
    "ImpactAssessment",
    "EscalationInfo",
    "DecisionContext",
    "DecisionContextBuilder",
    # Operational context
    "WorkShift",
    "StaffAvailability",
    "ProductionImpact",
    "OperationalContext",
    "OperationalContextBuilder",
    "adjust_severity_with_context",
]
