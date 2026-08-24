"""Baselines — Random Predictor."""

from __future__ import annotations

import random

from .prediction_signal import PredictionSignal
from .predictor import Predictor
from ..feature_window import FeatureWindow
from ..baselines.utils import clamp_prob


class RandomPredictor:
    """Moneda con ruido sembrado: cota honesta de "señal nula"."""

    def __init__(self, *, seed: int = 20260816, noise: float = 0.05) -> None:
        self._rng = random.Random(seed)
        self._noise = noise
        self.name = "random"

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        prob_up = clamp_prob(0.50 + self._rng.uniform(-self._noise, self._noise))
        return PredictionSignal(
            probability_up=prob_up,
            expected_return=0.0,
            lower=-1e-9,
            upper=1e-9,
            confidence_level=0.50,
        )