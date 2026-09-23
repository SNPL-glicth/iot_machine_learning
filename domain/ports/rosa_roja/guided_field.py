"""Guided Field domain port: Injected macro dynamics and prior knowledge.

Defines the hexagonal boundary contract for guiding trajectory generation
(e.g., Fourier macro velocity, Bayesian historical priors) without coupling
domain algorithms directly to external infrastructure models.
"""

from __future__ import annotations

from typing import Dict, Protocol, Tuple
import numpy as np


class GuidedFieldPort(Protocol):
    """Hexagonal domain port for querying global dynamics and historical prior distributions.

    Used by trajectory generators (e.g., RandomWalkSampler) to conduct Guided Importance
    Sampling when local memory buffers have zero trajectory density (dead ends).
    """

    def get_macro_gradient(self, timestamp: float) -> Tuple[np.ndarray, float]:
        """Calculates long-term macro trend vector and harmonic confidence.

        Args:
            timestamp: Epoch timestamp of current observation event.

        Returns:
            Tuple of:
                - np.ndarray: Normalized directional velocity vector in phase space.
                - float: Harmonic model confidence score in [0.0, 1.0].
        """
        ...

    def get_motif_prior(self, regime: str) -> Dict[int, float]:
        """Queries historical global prior probability distribution across topological motifs.

        Args:
            regime: Current detected macro regime name (e.g. 'stable', 'trending', 'volatile').

        Returns:
            Dict mapping motif_id to prior probability mass (summing to 1.0).
        """
        ...


class NullGuidedField(GuidedFieldPort):
    """No-op fallback implementation of GuidedFieldPort for isolated domain unit tests."""

    def get_macro_gradient(self, timestamp: float) -> Tuple[np.ndarray, float]:
        return np.zeros(1, dtype=np.float64), 0.0

    def get_motif_prior(self, regime: str) -> Dict[int, float]:
        return {}
