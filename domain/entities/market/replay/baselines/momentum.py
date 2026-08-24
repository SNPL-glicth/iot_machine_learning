"""Baselines — Momentum Predictor."""

from __future__ import annotations

import math

from .prediction_signal import PredictionSignal
from .predictor import Predictor
from ..feature_window import FeatureWindow
from ..baselines.utils import logistic, clamp_prob, band


class MomentumPredictor:
    """Dirección del drift de los últimos ``lookback`` retornos."""

    def __init__(self, *, scale: float = 2.0, lookback_override: int | None = None) -> None:
        self._scale = scale
        self._lookback_override = lookback_override
        self.name = "momentum"

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        lb = self._lookback_override or lookback
        if window.size < lb + 1:
            raise ValueError(f"ventana insuficiente para momentum ({lb + 1} velas)")
        drift = window.mean_return(lb)
        vol = window.std_return(lb) or 1e-12
        ratio = horizon_seconds / observation_interval
        prob_up = clamp_prob(logistic(drift / vol * math.sqrt(lb) * self._scale))
        expected = drift * ratio
        lower, upper = band(expected, vol, ratio)
        return PredictionSignal(
            probability_up=prob_up, expected_return=expected, lower=lower, upper=upper
        )