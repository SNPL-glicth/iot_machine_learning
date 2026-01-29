"""Módulo de contexto para decisiones ML.

REFACTORIZADO 2026-01-29:
- Modelos en models/
- Servicios en services/
"""

from .decision_context import (
    DecisionContextBuilder,
)
from .models import (
    WorkShift,
    StaffAvailability,
    ProductionImpact,
    OperationalContext,
    ActionUrgency,
    ImpactLevel,
    RecommendedAction,
    ImpactAssessment,
    EscalationInfo,
    DecisionContext,
)
from .operational_context import (
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
