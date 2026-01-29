"""Módulo de Contexto de Decisión para ML.

REFACTORIZADO 2026-01-29:
- Modelos extraídos a models/decision_models.py
- Servicios extraídos a services/decision_builder.py
- Este archivo ahora es solo el punto de entrada (~50 líneas, antes 516)

Estructura:
- models/decision_models.py: Dataclasses (RecommendedAction, ImpactAssessment, etc.)
- services/decision_builder.py: DecisionContextBuilder
"""

from __future__ import annotations

from sqlalchemy.engine import Connection

from .models.decision_models import DecisionContext
from .services.decision_builder import DecisionContextBuilder

# Re-exportar para compatibilidad
from .models.decision_models import (
    ActionUrgency,
    ImpactLevel,
    RecommendedAction,
    ImpactAssessment,
    EscalationInfo,
)

__all__ = [
    "DecisionContext",
    "DecisionContextBuilder",
    "ActionUrgency",
    "ImpactLevel",
    "RecommendedAction",
    "ImpactAssessment",
    "EscalationInfo",
]
