"""Performance metrics module for ML service.

Provides metrics collection for observability.
"""

from .performance_metrics import (
    MLMetrics,
    get_metrics,
    record_prediction,
    record_reading_processed,
)

__all__ = [
    "MLMetrics",
    "get_metrics",
    "record_prediction",
    "record_reading_processed",
]
