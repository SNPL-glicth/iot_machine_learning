"""Kalman engine adapter for Rosa Roja ExpertJury."""

from __future__ import annotations

import numpy as np
from infrastructure.ml.engines.rosa_roja.algorithms.domain.trajectory import Trajectory
from .base_adapter import BaseExpertAdapter
from infrastructure.ml.interfaces import PredictionEngine


class KalmanExpertAdapter(BaseExpertAdapter):
    """Adapter for Kalman Constant-Velocity engine."""

    def __init__(
        self,
        engine: PredictionEngine,
        is_critical: bool = False,
        threshold: float = 0.60,
        weight: float = 1.0,
    ):
        super().__init__(
            engine=engine,
            name="kalman",
            is_critical=is_critical,
            threshold=threshold,
            weight=weight,
        )

    def _trajectory_to_values(self, trajectory: Trajectory) -> np.ndarray:
        """Accumulate projected deltas into position levels for the 2D CV Kalman filter."""
        deltas = super()._trajectory_to_values(trajectory)
        return np.cumsum(deltas)


def create_kalman_adapter(**engine_kwargs) -> KalmanExpertAdapter:
    """Factory to create Kalman adapter with engine."""
    from infrastructure.ml.engines.kalman.engine import KalmanPredictionEngine
    engine = KalmanPredictionEngine(**engine_kwargs)
    return KalmanExpertAdapter(engine)