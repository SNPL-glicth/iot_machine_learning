"""Baselines — EMA Crossover Predictor."""

from __future__ import annotations

import math

from .prediction_signal import PredictionSignal
from .predictor import Predictor
from ..feature_window import FeatureWindow
from ..baselines.utils import logistic, clamp_prob, band


class EmaCrossoverPredictor:
    """Cruce de EMAs (rápida vs lenta) sobre closes de la ventana."""

    def __init__(self, *, fast: int = 4, slow: int = 12) -> None:
        if not 1 <= fast < slow:
            raise ValueError("se requiere 1 <= fast < slow")
        self._fast = fast
        self._slow = slow
        self.name = "ema-crossover"

    @staticmethod
    def _ema(values: tuple[float, ...], span: int) -> float:
        alpha = 2.0 / (span + 1.0)
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        return ema

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        closes = tuple(c.close for c in window.candles)
        if len(closes) < self._slow + 1:
            raise ValueError(
                f"ventana insuficiente para ema-crossover ({self._slow} velas)"
            )
        fast_ema = self._ema(closes, self._fast)
        slow_ema = self._ema(closes, self._slow)
        if slow_ema <= 0:
            raise ValueError("closes deben ser > 0 para ema-crossover")
        spread = (fast_ema - slow_ema) / slow_ema
        ratio = horizon_seconds / observation_interval
        prob_up = clamp_prob(logistic(spread * 40.0))
        expected = spread * ratio * 0.5
        vol = window.std_return(self._slow) or 1e-12
        lower, upper = band(expected, vol, ratio)
        return PredictionSignal(
            probability_up=prob_up, expected_return=expected, lower=lower, upper=upper
        )