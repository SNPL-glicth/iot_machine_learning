"""Entidades de resultados de inferencia.

Anomalías, predicciones y resultados de búsqueda en memoria cognitiva.
"""

from .anomaly import AnomalyResult, AnomalySeverity
from .prediction import PredictionConfidence, Prediction
from .memory_search_result import MemorySearchResult

__all__ = [
    "AnomalyResult",
    "AnomalySeverity",
    "Prediction",
    "PredictionConfidence",
    "MemorySearchResult",
]
