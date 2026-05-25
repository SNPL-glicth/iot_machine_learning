"""Rolling Z-score sub-detector — detección de drift por ventana corta vs larga móvil.

Compara la media de una ventana corta reciente contra la media de una
ventana larga móvil (últimos N puntos). Detecta desplazamientos de régimen
(gradual drift) que los métodos globales fijos pierden.

Sin sklearn, sin I/O.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import List, Optional

from core.parameters.numerical_constants import STAT_THRESHOLDS, EPSILON

from ..core.protocol import SubDetector
from ..scoring.functions import compute_z_score, compute_z_vote

logger = logging.getLogger(__name__)


class RollingZScoreDetector(SubDetector):
    """Sub-detector basado en Z-score de ventana corta vs larga móvil.

    Atributos:
        _short_window: Tamaño de la ventana corta (puntos recientes).
        _long_window: Tamaño de la ventana larga móvil.
        _lower: Z-score debajo del cual el voto es 0.
        _upper: Z-score encima del cual el voto es 1.
        _value_history: Todos los valores vistos (para ventana larga móvil).
    """

    def __init__(
        self,
        short_window: int = 10,  # v1.0 production config
        long_window: int = 400,  # v1.0 production config
        lower: float = 3.5,  # v1.0 production config
        upper: float = 3.5,  # v1.0 production config
        hysteresis: int = 3,  # v1.0 production config
    ) -> None:

        self._short_window = short_window
        self._long_window = long_window
        self._lower = lower
        self._upper = upper
        self._hysteresis = max(1, hysteresis)
        self._value_history: deque[float] = deque(maxlen=500)
        self._consecutive_count: int = 0
        self._audit_z_scores: list[float] = []

        logger.info(f"RollingZ init: long={long_window}, hyst={hysteresis}, z={upper}")

    @property
    def method_name(self) -> str:
        return "rolling_z"

    def train(self, values: List[float], **kwargs: object) -> None:
        n = len(values)
        if n < self._long_window:
            return
        self._value_history.clear()
        self._value_history.extend(values)

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        self._value_history.append(value)

        if len(self._value_history) < self._long_window:
            return None

        # Long window = últimos _long_window puntos (móvil)
        long_values = list(self._value_history)[-self._long_window:]
        long_mean = sum(long_values) / len(long_values)
        variance = sum((v - long_mean) ** 2 for v in long_values) / max(len(long_values) - 1, 1)
        long_std = math.sqrt(variance)

        # Short window = últimos _short_window puntos
        short_values = long_values[-self._short_window:]
        short_mean = sum(short_values) / len(short_values)

        # Standard error of the mean for the short window
        long_std = long_std if long_std > EPSILON.DIVISION else EPSILON.DIVISION
        sem = long_std / math.sqrt(self._short_window)

        # Z-score: how many standard errors is short_mean from long_mean?
        z = compute_z_score(short_mean, long_mean, sem)

        raw_vote = compute_z_vote(z, self._lower, self._upper)

        # Hysteresis counter tracks persistence but does NOT zero-out the vote.
        # Output is now continuous: proportional to z-score distance from threshold.
        if raw_vote > 0:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0

        result = raw_vote if self._consecutive_count >= self._hysteresis else 0.0
        self._audit_z_scores.append(float(z))

        logger.debug(
            "rolling_z_vote",
            extra={
                "value": value,
                "short_mean": round(short_mean, 4),
                "long_mean": round(long_mean, 4),
                "sem": round(sem, 4),
                "z": round(z, 4),
                "raw_vote": round(raw_vote, 4),
                "consecutive": self._consecutive_count,
                "hysteresis": self._hysteresis,
                "vote": round(result, 4),
            },
        )

        return result

    @property
    def is_trained(self) -> bool:
        return len(self._value_history) >= self._long_window
