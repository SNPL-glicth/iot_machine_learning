from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Trend = Literal["up", "down", "stable"]


@dataclass(frozen=True)
class RegressionModel:
    """Modelo de regresión lineal sencillo por sensor.

    Representa y = intercept_ + coef_ * t, donde t son minutos desde el primer dato.
    """

    sensor_id: int
    coef_: float
    intercept_: float
    r2: float
    horizon_minutes: int


def compute_trend(coef: float, eps: float = 1e-3) -> Trend:
    if coef > eps:
        return "up"
    if coef < -eps:
        return "down"
    return "stable"