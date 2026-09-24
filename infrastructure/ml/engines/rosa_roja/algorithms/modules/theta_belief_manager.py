"""ThetaBeliefManager: Handles ThetaBelief posterior updates for RhythmGenerator."""

from __future__ import annotations

from typing import Optional
import numpy as np

from domain.entities.rosa_roja.movement import Movement
from domain.entities.rosa_roja.theta_belief import StateKey, ThetaBelief
from .voronoi_classifier import VoronoiMotifClassifier, DEFAULT_N_CENTROIDS


class ThetaBeliefManager:
    """Manages ThetaBelief updates from movement history using Voronoi vector quantization."""

    def __init__(
        self,
        theta_alpha: float = 0.95,
        quantization_decimals: int = 2,
        motif_classifier: Optional[VoronoiMotifClassifier] = None,
        n_centroids: int = DEFAULT_N_CENTROIDS,
    ):
        self._theta = ThetaBelief(alpha=theta_alpha)
        self._quantization_decimals = quantization_decimals
        self._motif_classifier = motif_classifier or VoronoiMotifClassifier(n_centroids=n_centroids)

    def _quantize_state(self, state: np.ndarray) -> StateKey:
        return self._motif_classifier.classify(state)

    def update_from_history(self, history: list[Movement]) -> None:
        """Update belief from full history (initial) or latest transition (incremental)."""
        if len(history) < 2:
            return

        if self._theta.total_updates == 0:
            # Initial: replay full history
            for curr, nxt in zip(history, history[1:]):
                from_key = self._quantize_state(curr.delta_state)
                to_key = self._quantize_state(nxt.delta_state)
                self._theta.update(from_key, to_key)
        else:
            # Incremental: only latest transition
            prev = history[-2]
            latest = history[-1]
            self._theta.update(
                self._quantize_state(prev.delta_state),
                self._quantize_state(latest.delta_state),
            )

    def compute_entropy(self, current_state_key: Optional[StateKey]) -> float:
        """Compute conditional entropy H(Θ|D_t)."""
        return self._theta.compute_entropy(current_state_key)

    def get_transition_probabilities(self, state_key: StateKey) -> dict:
        """Get posterior transition probabilities for a state."""
        return self._theta.get_transition_probabilities(state_key)

    def reset(self) -> None:
        self._theta.reset()
        self._motif_classifier.reset()

    @property
    def theta(self) -> ThetaBelief:
        return self._theta