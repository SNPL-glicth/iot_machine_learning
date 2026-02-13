"""Z-score sub-detector — detección por desviación estándar de magnitud.

Una responsabilidad: evaluar si un valor está lejos de la media histórica.
Sin sklearn, sin I/O.
"""

from __future__ import annotations

from typing import List, Optional

from ..detector_protocol import SubDetector
from ..scoring_functions import compute_z_score, compute_z_vote
from ..training_stats import TrainingStats, compute_training_stats


class ZScoreDetector(SubDetector):
    """Sub-detector basado en Z-score de magnitud.

    Attributes:
        _lower: Z-score debajo del cual el voto es 0.
        _upper: Z-score encima del cual el voto es 1.
    """

    def __init__(self, lower: float = 2.0, upper: float = 3.0) -> None:
        self._lower = lower
        self._upper = upper
        self._stats: Optional[TrainingStats] = None

    @property
    def method_name(self) -> str:
        return "z_score"

    def train(self, values: List[float], **kwargs: object) -> None:
        self._stats = compute_training_stats(values)

    def vote(self, value: float, **kwargs: object) -> Optional[float]:
        if self._stats is None:
            return None
        z = compute_z_score(value, self._stats.mean, self._stats.std)
        return compute_z_vote(z, self._lower, self._upper)

    @property
    def is_trained(self) -> bool:
        return self._stats is not None

    @property
    def last_z_score(self) -> float:
        """Último Z-score calculado (para narración). Recalcula bajo demanda."""
        return 0.0
