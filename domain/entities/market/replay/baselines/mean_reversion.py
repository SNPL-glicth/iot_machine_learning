"""Baselines — Mean Reversion Predictor."""

from __future__ import annotations

import math

from .prediction_signal import PredictionSignal
from .predictor import Predictor
from ..feature_window import FeatureWindow
from ..baselines.utils import logistic, clamp_prob, band


class MeanReversionPredictor:
    """Reversión a la media: el precio lejos de su EMA lenta tiende a volver."""

    def __init__(self, *, span: int = 12) -> None:
        self._span = span
        self.name = "mean-reversion"

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        closes = tuple(c.close for c in window.candles)
        if len(closes) < self._span + 1:
            raise ValueError(
                f"ventana insuficiente para mean-reversion ({self._span} velas)"
            )
        alpha = 2.0 / (self._span + 1.0)
        ema = closes[0]
        for value in closes[1:]:
            ema = alpha * value + (1 - alpha) * ema
        if ema <= 0:
            raise ValueError("ema debe ser > 0 para mean-reversion")
        deviation = (closes[-1] - ema) / ema
        ratio = horizon_seconds / observation_interval
        prob_up = clamp_prob(logistic(-deviation * 40.0))
        expected = -deviation * ratio * 0.5
        vol = window.std_return(self._span) or 1e-12
        lower, upper = band(expected, vol, ratio)
        return PredictionSignal(
            probability_up=prob_up, expected_return=expected, lower=lower, upper=upper
        )