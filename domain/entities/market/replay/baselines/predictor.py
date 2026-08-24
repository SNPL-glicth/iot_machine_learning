"""Baselines — Predictor Protocol."""

from __future__ import annotations

from typing import Protocol

from ..feature_window import FeatureWindow
from .prediction_signal import PredictionSignal


class Predictor(Protocol):
    """Contrato de un predictor evaluable por el benchmark."""

    name: str

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        ...