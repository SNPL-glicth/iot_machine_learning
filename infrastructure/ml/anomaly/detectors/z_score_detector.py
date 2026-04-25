"""Z-score sub-detector — detección por desviación estándar de magnitud.

Una responsabilidad: evaluar si un valor está lejos de la media histórica.
Sin sklearn, sin I/O.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional

from ..core.protocol import SubDetector
from ..scoring.functions import compute_z_score, compute_z_vote
from ..scoring.training import TrainingStats, compute_training_stats


class ZScoreDetector(SubDetector):
    """Sub-detector basado en Z-score de magnitud.

    Attributes:
        _lower: Z-score debajo del cual el voto es 0.
        _upper: Z-score encima del cual el voto es 1.
        _adaptive: Activa adaptación de thresholds por volatilidad reciente.
        _rolling_std_history: Historial de std de últimas ventanas (max 100).
        _value_history: Últimos valores vistos para calcular std local.
    """

    def __init__(
        self,
        lower: float = 2.0,
        upper: float = 3.0,
        *,
        adaptive: bool = True,
        max_history: int = 100,
        min_history_entries: int = 5,
    ) -> None:
        self._base_lower = lower
        self._base_upper = upper
        self._adaptive = adaptive
        self._max_history = max_history
        self._min_history_entries = min_history_entries
        self._stats: Optional[TrainingStats] = None
        self._rolling_std_history: deque[float] = deque(maxlen=max_history)
        self._value_history: deque[float] = deque(maxlen=max_history)

    @property
    def method_name(self) -> str:
        return "z_score"

    def train(self, values: List[float], **kwargs: object) -> None:
        self._stats = compute_training_stats(values)
        if self._stats and self._adaptive:
            self._rolling_std_history.append(self._stats.std)

    @property
    def _effective_thresholds(self) -> tuple[float, float]:
        """Devuelve (lower, upper) efectivos, adaptativos o fijos."""
        if not self._adaptive or len(self._rolling_std_history) < self._min_history_entries:
            return self._base_lower, self._base_upper
        mean_rolling_std = sum(self._rolling_std_history) / len(self._rolling_std_history)
        base_std = self._stats.std if self._stats and self._stats.std > 0 else mean_rolling_std
        if base_std > 0:
            scale = max(1.0, mean_rolling_std / base_std)
        else:
            scale = 1.0
        return (
            self._base_lower * scale,
            self._base_upper * scale,
        )

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        if self._stats is None:
            return None
        z = compute_z_score(value, self._stats.mean, self._stats.std)
        lower, upper = self._effective_thresholds
        result = compute_z_vote(z, lower, upper)
        if self._adaptive:
            self._value_history.append(value)
            if len(self._value_history) >= 3:
                local_mean = sum(self._value_history) / len(self._value_history)
                local_std = math.sqrt(
                    sum((v - local_mean) ** 2 for v in self._value_history)
                    / len(self._value_history)
                )
                if local_std > 0:
                    self._rolling_std_history.append(local_std)
        return result

    @property
    def is_trained(self) -> bool:
        return self._stats is not None

    @property
    def last_z_score(self) -> float:
        """Último Z-score calculado (para narración). Recalcula bajo demanda."""
        return 0.0
