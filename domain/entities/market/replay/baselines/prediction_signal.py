"""Baselines — PredictionSignal."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PredictionSignal:
    """Señal mínima que un predictor entrega al engine."""

    probability_up: float
    expected_return: float
    lower: float
    upper: float
    confidence_level: float = 0.50

    def __post_init__(self) -> None:
        if not 0.05 <= self.probability_up <= 0.95:
            raise ValueError(
                f"probability_up fuera de [0.05, 0.95]: {self.probability_up!r}"
            )
        if not math.isfinite(self.expected_return):
            raise ValueError("expected_return debe ser finito")
        if not self.lower <= self.expected_return <= self.upper:
            raise ValueError(
                "expected_return debe estar contenido en [lower, upper]"
            )