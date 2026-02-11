"""Motor baseline: media móvil simple.

Migrado desde ml/baseline.py + ml/metadata.py + engine_factory.BaselineMovingAverageEngine.
Responsabilidad ÚNICA: predicción por media móvil.
Sin dependencias externas, sin I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class BaselineConfig:
    """Config simple para baseline de media móvil.

    Attributes:
        window: Número máximo de puntos recientes a usar para la media.
    """

    window: int = 20


@dataclass(frozen=True)
class BaselineMetadata:
    """Metadatos para el modelo baseline.

    Se almacena en la tabla ``ml_models``.
    """

    name: str
    model_type: str
    version: str


BASELINE_MOVING_AVERAGE = BaselineMetadata(
    name="baseline_moving_average",
    model_type="baseline",
    version="1.0.0",
)


def predict_moving_average(
    values: Iterable[float], cfg: BaselineConfig
) -> Tuple[float, float]:
    """Devuelve (predicted_value, confidence) usando media móvil simple.

    - predicted_value: media de los últimos ``cfg.window`` valores.
    - confidence: ratio entre nº de puntos usados y ``cfg.window`` (máx 1.0).
    """
    seq = list(values)
    if not seq:
        return 0.0, 0.0

    window = max(1, cfg.window)
    recent = seq[-window:]
    predicted = float(mean(recent))
    confidence = min(1.0, len(recent) / float(window))

    return predicted, confidence
