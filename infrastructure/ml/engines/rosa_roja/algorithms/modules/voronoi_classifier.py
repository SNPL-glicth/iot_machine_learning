"""VoronoiMotifClassifier: Online phase-space vector quantization into K topological motifs."""

from __future__ import annotations

import logging
from typing import Optional, Sequence
import numpy as np

from domain.entities.rosa_roja.theta_belief import StateKey

logger = logging.getLogger(__name__)

DEFAULT_N_CENTROIDS: int = 8  # K in [6, 12]
DEFAULT_LEARNING_RATE: float = 0.05
DEFAULT_MIN_DISTANCE: float = 0.05


class VoronoiMotifClassifier:
    """Classifies continuous kinematic state vectors into K Voronoi cells / motifs in [0, K-1].

    Replaces rigid Cartesian decimal rounding with scale-invariant topological Voronoi
    vector quantization. Up to K centroids adapt smoothly via online k-means (MacQueen style).
    Points within a Voronoi basin of attraction map to the canonical centroid codevector.
    """

    def __init__(
        self,
        n_centroids: int = DEFAULT_N_CENTROIDS,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        quantization_decimals: int = 2,
        min_distance: float = DEFAULT_MIN_DISTANCE,
    ) -> None:
        if not (6 <= n_centroids <= 12):
            raise ValueError(f"n_centroids must be in [6, 12], got {n_centroids}")
        self.n_centroids = n_centroids
        self.learning_rate = learning_rate
        self.quantization_decimals = quantization_decimals
        self.min_distance = min_distance
        self._centroids: list[np.ndarray] = []
        self._counts: list[int] = []
        self._centroids_arr: Optional[np.ndarray] = None
        self._centroid_keys: list[StateKey] = []

    @property
    def centroids(self) -> list[np.ndarray]:
        return self._centroids

    def _sync(self) -> None:
        """Synchronizes numpy matrix cache and precomputed StateKey tuples."""
        if self._centroids:
            self._centroids_arr = np.array(self._centroids, dtype=np.float64)
            self._centroid_keys = [self._to_key(c) for c in self._centroids]
        else:
            self._centroids_arr = None
            self._centroid_keys.clear()

    def learn_sample(self, state: Sequence[float] | np.ndarray) -> None:
        """Adapts centroids on a newly observed data point (MacQueen online k-means)."""
        arr = np.asarray(state, dtype=np.float64).flatten()
        if arr.size == 0:
            return

        if not self._centroids:
            self._centroids.append(arr.copy())
            self._counts.append(1)
            self._sync()
            return

        assert self._centroids_arr is not None
        diff = self._centroids_arr - arr
        sq_dists = np.sum(diff * diff, axis=1)
        min_idx = int(np.argmin(sq_dists))
        min_dist = float(np.sqrt(sq_dists[min_idx]))

        if len(self._centroids) < self.n_centroids and min_dist > self.min_distance:
            self._centroids.append(arr.copy())
            self._counts.append(1)
            self._sync()
            return

        self._counts[min_idx] += 1
        eta = self.learning_rate / np.sqrt(1.0 + 0.01 * self._counts[min_idx])
        self._centroids[min_idx] += eta * (arr - self._centroids[min_idx])
        self._sync()

    def classify(self, state: Sequence[float] | np.ndarray, update: bool = False) -> StateKey:
        """Assigns continuous state to nearest Voronoi centroid codevector without moving centroids."""
        if update:
            self.learn_sample(state)

        arr = np.asarray(state, dtype=np.float64).flatten()
        if arr.size == 0:
            return ()

        if not self._centroids:
            self.learn_sample(arr)
            return self._centroid_keys[0]

        assert self._centroids_arr is not None
        diff = self._centroids_arr - arr
        sq_dists = np.sum(diff * diff, axis=1)
        min_idx = int(np.argmin(sq_dists))
        min_dist = float(np.sqrt(sq_dists[min_idx]))

        if len(self._centroids) < self.n_centroids and min_dist > self.min_distance:
            self._centroids.append(arr.copy())
            self._counts.append(1)
            self._sync()
            return self._centroid_keys[-1]

        return self._centroid_keys[min_idx]

    def quantize(self, state: Sequence[float] | np.ndarray) -> StateKey:
        """Alias for classify() to satisfy quantize interface."""
        return self.classify(state, update=False)

    def get_motif_id(self, state: Sequence[float] | np.ndarray) -> int:
        """Returns integer index [0, K-1] of the nearest Voronoi centroid."""
        arr = np.asarray(state, dtype=np.float64).flatten()
        if not self._centroids or arr.size == 0:
            return 0
        assert self._centroids_arr is not None
        diff = self._centroids_arr - arr
        return int(np.argmin(np.sum(diff * diff, axis=1)))

    def _to_key(self, centroid: np.ndarray) -> StateKey:
        return tuple(round(float(v), self.quantization_decimals) for v in centroid)

    def reset(self) -> None:
        """Resets centroids and counts to uninitialized state."""
        self._centroids.clear()
        self._counts.clear()
        self._centroids_arr = None
        self._centroid_keys.clear()
