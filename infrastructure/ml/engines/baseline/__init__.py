"""Baseline prediction engine — simple moving average.

Components:
    - predict_moving_average: Pure function for moving average prediction
    - BaselineConfig: Configuration dataclass
    - BaselinePredictionAdapter: DEPRECATED - Adapter wrapping as PredictionPort
"""

from .engine import (
    BASELINE_MOVING_AVERAGE,
    BaselineConfig,
    BaselineMetadata,
    predict_moving_average,
)
from .adapter import BaselinePredictionAdapter

__all__ = [
    "predict_moving_average",
    "BaselineConfig",
    "BaselineMetadata",
    "BASELINE_MOVING_AVERAGE",
    "BaselinePredictionAdapter",
]
