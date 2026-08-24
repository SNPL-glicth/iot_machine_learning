"""Adaptación controlada de ZENIN (FASE 8).

Proposal ≠ Update: el sistema primero propone (con razón auditable) y
solo aplica si el guardrail acepta, versionando el modelo (append-only).

    Historial (solo outcomes reales)
        → PerformanceAnalyzer → ExpertScores
        → WeightProposer → WeightProposal (no toca nada)
        → AdaptationGuard → ACCEPT / REJECT
        → model_versions (v2 = v1 + propuestas aceptadas)

Regla de piedra: ZENIN nunca aprende de su propia predicción sin haber
observado el outcome externo.
"""

from .expert_scores import ExpertScore, PerformanceAnalyzer
from .guard import AdaptationGuard, GuardCheck, GuardResult, wilson_lower_bound
from .proposer import WeightProposal, WeightProposer, default_weights
from .selection import (
    ExpertNetScore,
    SelectionConfig,
    SelectionMode,
    SelectionResult,
    expert_net_scores,
    select_weights,
)

__all__ = [
    "ExpertScore",
    "PerformanceAnalyzer",
    "AdaptationGuard",
    "GuardCheck",
    "GuardResult",
    "wilson_lower_bound",
    "WeightProposal",
    "WeightProposer",
    "default_weights",
    "SelectionMode",
    "SelectionConfig",
    "ExpertNetScore",
    "expert_net_scores",
    "SelectionResult",
    "select_weights",
]
