"""Baseline prediction engine — simple moving average.

Components:
    - predict_moving_average: Pure function for moving average prediction
    - BaselineConfig: Configuration dataclass

Note: BaselinePredictionAdapter was deprecated and removed.
Use EngineFactory.create("baseline_moving_average").as_port() instead.
"""

from .engine import (
    BASELINE_MOVING_AVERAGE,
    BaselineConfig,
    BaselineMetadata,
    predict_moving_average,
)

__all__ = [
    "predict_moving_average",
    "BaselineConfig",
    "BaselineMetadata",
    "BASELINE_MOVING_AVERAGE",
]
