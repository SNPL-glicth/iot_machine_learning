"""IQR sub-detector — detección por rango intercuartílico.

Una responsabilidad: evaluar si un valor está fuera de los Tukey fences.
Sin sklearn, sin I/O.
"""

from __future__ import annotations

import math
from collections import deque
from typing import List, Optional

from ..core.protocol import SubDetector
from ..scoring.functions import compute_iqr_vote
from ..scoring.training import TrainingStats, compute_training_stats


class IQRDetector(SubDetector):
    """Sub-detector basado en IQR (Tukey fences).

    Attributes:
        _adaptive: Activa adaptación de fences por volatilidad reciente.
        _rolling_iqr_history: Historial de IQR de últimas ventanas (max 100).
        _value_history: Últimos valores vistos para calcular IQR local.
    """

    def __init__(
        self,
        *,
        adaptive: bool = True,
        max_history: int = 100,
        min_history_entries: int = 5,
    ) -> None:
        self._adaptive = adaptive
        self._max_history = max_history
        self._min_history_entries = min_history_entries
        self._stats: Optional[TrainingStats] = None
        self._rolling_iqr_history: deque[float] = deque(maxlen=max_history)
        self._value_history: deque[float] = deque(maxlen=max_history)

    @property
    def method_name(self) -> str:
        return "iqr"

    def train(self, values: List[float], **kwargs: object) -> None:
        self._stats = compute_training_stats(values)
        if self._stats and self._adaptive:
            self._rolling_iqr_history.append(self._stats.iqr)

    @property
    def _effective_fence_multiplier(self) -> float:
        """Devuelve multiplicador de fences adaptativo o fijo (1.5)."""
        if not self._adaptive or len(self._rolling_iqr_history) < self._min_history_entries:
            return 1.5
        mean_rolling_iqr = sum(self._rolling_iqr_history) / len(self._rolling_iqr_history)
        base_iqr = self._stats.iqr if self._stats and self._stats.iqr > 0 else mean_rolling_iqr
        if base_iqr > 0:
            scale = max(1.0, mean_rolling_iqr / base_iqr)
        else:
            scale = 1.0
        return min(3.0, 1.5 * scale)

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        if self._stats is None:
            return None
        multiplier = self._effective_fence_multiplier
        q1 = self._stats.q1
        q3 = self._stats.q3
        iqr = self._stats.iqr
        if self._adaptive:
            self._value_history.append(value)
            if len(self._value_history) >= 4:
                sorted_vals = sorted(self._value_history)
                n = len(sorted_vals)
                q1_idx = int(n * 0.25)
                q3_idx = int(n * 0.75)
                local_q1 = sorted_vals[q1_idx]
                local_q3 = sorted_vals[q3_idx]
                local_iqr = local_q3 - local_q1
                if local_iqr > 0:
                    self._rolling_iqr_history.append(local_iqr)
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        return 1.0 if (value < lower or value > upper) else 0.0

    @property
    def is_trained(self) -> bool:
        return self._stats is not None
