from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Tuple


@dataclass(frozen=True)
class BaselineConfig:
    """Config simple para baseline de media móvil.

    Attributes
    ----------
    window: int
        Número máximo de puntos recientes a usar para la media.
    """

    window: int = 20


def predict_moving_average(values: Iterable[float], cfg: BaselineConfig) -> Tuple[float, float]:
    """Devuelve (predicted_value, confidence) usando una media móvil muy simple.

    - predicted_value: media de los últimos ``cfg.window`` valores.
    - confidence: ratio entre nº de puntos usados y ``cfg.window`` (máx 1.0).
    """

    seq = list(values)
    if not seq:
        # Sin datos: devolvemos 0 con confianza 0.0 para no romper el flujo.
        return 0.0, 0.0

    window = max(1, cfg.window)
    recent = seq[-window:]
    predicted = float(mean(recent))

    confidence = min(1.0, len(recent) / float(window))

    return predicted, confidence
