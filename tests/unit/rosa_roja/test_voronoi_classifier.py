"""Unit tests for VoronoiMotifClassifier and single source of truth quantization."""

from __future__ import annotations

import numpy as np
import pytest

from infrastructure.ml.engines.rosa_roja.algorithms.modules.voronoi_classifier import (
    VoronoiMotifClassifier,
    DEFAULT_N_CENTROIDS,
)
from infrastructure.ml.engines.rosa_roja.algorithms.modules.rhythm_generator import (
    RhythmTrajectoryGenerator,
)
from infrastructure.ml.engines.rosa_roja.algorithms.modules.theta_belief_manager import (
    ThetaBeliefManager,
)


def test_voronoi_centroid_bounds_validation():
    """Centroid count must be constrained to K in [6, 12]."""
    assert DEFAULT_N_CENTROIDS == 8
    assert 6 <= DEFAULT_N_CENTROIDS <= 12

    # Valid boundaries
    clf_6 = VoronoiMotifClassifier(n_centroids=6)
    assert clf_6.n_centroids == 6
    clf_12 = VoronoiMotifClassifier(n_centroids=12)
    assert clf_12.n_centroids == 12

    # Invalid below 6 or above 12
    with pytest.raises(ValueError, match=r"n_centroids must be in \[6, 12\]"):
        VoronoiMotifClassifier(n_centroids=5)

    with pytest.raises(ValueError, match=r"n_centroids must be in \[6, 12\]"):
        VoronoiMotifClassifier(n_centroids=13)


def test_voronoi_motif_classification_and_adaptation():
    """Voronoi classifier allocates up to K centroids and maps to closest basin."""
    clf = VoronoiMotifClassifier(n_centroids=8, min_distance=0.1)

    # First point initializes centroid 0
    k0 = clf.classify(np.array([1.0, 0.0]), update=True)
    assert k0 == (1.0, 0.0)
    assert len(clf.centroids) == 1
    assert clf.get_motif_id(np.array([1.0, 0.0])) == 0

    # Distant point initializes centroid 1
    k1 = clf.classify(np.array([5.0, 0.0]), update=True)
    assert k1 == (5.0, 0.0)
    assert len(clf.centroids) == 2
    assert clf.get_motif_id(np.array([5.0, 0.0])) == 1

    # Nearby point to centroid 0 maps to centroid 0
    k_near0 = clf.classify(np.array([1.02, 0.01]), update=False)
    assert k_near0 == (1.0, 0.0)
    assert clf.get_motif_id(np.array([1.02, 0.01])) == 0

    # Nearby point to centroid 1 maps to centroid 1
    k_near1 = clf.classify(np.array([4.98, -0.02]), update=False)
    assert k_near1 == (5.0, 0.0)
    assert clf.get_motif_id(np.array([4.98, -0.02])) == 1


def test_rhythm_and_theta_share_single_motif_source_of_truth():
    """_quantize_state in rhythm_generator and theta_belief_manager share the identical classifier."""
    rhythm = RhythmTrajectoryGenerator(n_centroids=8)
    theta_mgr = rhythm._theta_manager

    # Must share the exact same VoronoiMotifClassifier instance
    assert rhythm._motif_classifier is theta_mgr._motif_classifier

    # State quantization returns identical keys from both modules
    test_vec = np.array([3.1415, -2.7182])
    k_rhythm = rhythm._quantize_state(test_vec)
    k_theta = theta_mgr._quantize_state(test_vec)
    assert k_rhythm == k_theta
