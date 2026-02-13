"""Estrategia de voting desacoplada para ensemble de anomalías.

Una responsabilidad: combinar votos de sub-detectores en un score final.
Sin sklearn, sin I/O, sin estado de entrenamiento.
"""

from __future__ import annotations

from typing import Dict

from .scoring_functions import compute_consensus_confidence, weighted_vote


class VotingStrategy:
    """Combina votos de sub-detectores usando promedio ponderado.

    Attributes:
        _weights: Pesos por método.
        _threshold: Score > threshold → anomalía.
        _default_weight: Peso para métodos sin peso definido.
    """

    def __init__(
        self,
        weights: Dict[str, float],
        threshold: float = 0.5,
        default_weight: float = 0.1,
    ) -> None:
        self._weights = dict(weights)
        self._threshold = threshold
        self._default_weight = default_weight

    def combine(self, votes: Dict[str, float]) -> float:
        """Calcula score final a partir de votos individuales.

        Args:
            votes: Dict método → voto [0, 1].

        Returns:
            Score final en [0, 1].
        """
        return weighted_vote(votes, self._weights, self._default_weight)

    def is_anomaly(self, score: float) -> bool:
        """Determina si el score indica anomalía.

        Args:
            score: Score combinado.

        Returns:
            ``True`` si score > threshold.
        """
        return score > self._threshold

    def confidence(self, votes: Dict[str, float]) -> float:
        """Calcula confianza basada en consenso entre votantes.

        Args:
            votes: Dict método → voto [0, 1].

        Returns:
            Confianza en [0.5, 1.0].
        """
        return compute_consensus_confidence(votes)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)
