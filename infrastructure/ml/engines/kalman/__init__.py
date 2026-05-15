"""Kalman prediction engine package.

Submodules:
- ``kalman_cv_math`` — Pure 2D constant-velocity Kalman math (numpy-based).
- ``engine`` — KalmanPredictionEngine implementing PredictionEngine.
"""

from __future__ import annotations

from .kalman_cv_math import KalmanCVState, initialize_cv_state, predict_cv, update_cv
from .engine import KalmanPredictionEngine

__all__ = [
    "KalmanCVState",
    "initialize_cv_state",
    "predict_cv",
    "update_cv",
    "KalmanPredictionEngine",
]
