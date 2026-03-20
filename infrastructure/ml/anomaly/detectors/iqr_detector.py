"""IQR sub-detector — detección por rango intercuartílico.

Una responsabilidad: evaluar si un valor está fuera de los Tukey fences.
Sin sklearn, sin I/O.
"""

from __future__ import annotations

from typing import List, Optional

from ..core.protocol import SubDetector
from ..scoring.functions import compute_iqr_vote
from ..scoring.training import TrainingStats, compute_training_stats


class IQRDetector(SubDetector):
    """Sub-detector basado en IQR (Tukey fences)."""

    def __init__(self) -> None:
        self._stats: Optional[TrainingStats] = None

    @property
    def method_name(self) -> str:
        return "iqr"

    def train(self, values: List[float], **kwargs: object) -> None:
        self._stats = compute_training_stats(values)

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        if self._stats is None:
            return None
        return compute_iqr_vote(
            value, self._stats.q1, self._stats.q3, self._stats.iqr
        )

    @property
    def is_trained(self) -> bool:
        return self._stats is not None
