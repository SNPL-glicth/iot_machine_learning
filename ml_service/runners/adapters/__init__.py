"""Adapters for batch runner enterprise bridge.

Connects the batch runner loop to the enterprise ML stack
(PredictSensorValueUseCase, PredictionDomainService) without
modifying the existing prediction/event writers.
"""

from .enterprise_prediction import EnterprisePredictionAdapter, BatchPredictionResult
from .fallback_baseline import fallback_to_baseline

__all__ = [
    "EnterprisePredictionAdapter",
    "BatchPredictionResult",
    "fallback_to_baseline",
]
