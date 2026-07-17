"""Estrategia de voting para ensemble de anomalías.
Una responsabilidad: combinar votos en un score final.
"""
from __future__ import annotations
from typing import Dict, Optional

from ..scoring.functions import weighted_vote, compute_consensus_confidence


class VotingStrategy:

    def __init__(
        self,
        weights: Dict[str, float],
        threshold: float = 0.75,
        default_weight: float = 0.1,
    ) -> None:
        self._weights = dict(weights)
        self._threshold = threshold
        self._default_weight = default_weight

    def combine(self, votes: Dict[str, Optional[float]]) -> float:
        return weighted_vote(
            votes,
            self._weights,
            self._default_weight,
        )

    def is_anomaly(self, score: float, votes: Optional[Dict[str, Optional[float]]] = None) -> bool:
        if votes is not None:
            fired_weight = sum(
                self._weights[m] for m, v in votes.items()
                if m in self._weights and v is not None and v > 0.0
            )
            effective_threshold = max(self._threshold * fired_weight, self._threshold * 0.6)
            return score >= effective_threshold
        return score >= self._threshold

    def confidence(self, votes: Dict[str, Optional[float]]) -> float:
        return compute_consensus_confidence(votes)

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights)
