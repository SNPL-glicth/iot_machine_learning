"""Módulos de cognición, evaluación post-mortem y meta-competencia."""

from .failure_taxonomy import (
    FailureDiagnostic,
    FailureReason,
    diagnose_prediction_failure,
)
from .post_mortem_evaluator import (
    PendingPrediction,
    PostMortemEvaluator,
    PostMortemRecord,
)
from .metacognitive_tracker import (
    MetacognitiveStatus,
    MetacognitiveTracker,
)

__all__ = [
    "FailureReason",
    "FailureDiagnostic",
    "diagnose_prediction_failure",
    "PendingPrediction",
    "PostMortemRecord",
    "PostMortemEvaluator",
    "MetacognitiveStatus",
    "MetacognitiveTracker",
]
