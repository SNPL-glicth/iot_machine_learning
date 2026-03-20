"""Statistical prediction engine — EMA/Holt-based forecasting.

Components:
    - StatisticalPredictionEngine: Double exponential smoothing (Holt's method)
"""

from .engine import StatisticalPredictionEngine

__all__ = ["StatisticalPredictionEngine"]
