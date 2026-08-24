"""Baselines — Naive Predictor."""

from __future__ import annotations

from .prediction_signal import PredictionSignal
from .predictor import Predictor
from ..feature_window import FeatureWindow


class NaivePredictor:
    """Mañana = último precio (martingala). Sin señal direccional."""

    name = "naive"

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        return PredictionSignal(
            probability_up=0.50,
            expected_return=0.0,
            lower=-1e-9,
            upper=1e-9,
            confidence_level=0.50,
        )