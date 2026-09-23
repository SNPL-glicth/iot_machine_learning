"""Unit tests for TopologicalMotifKey, GuidedFieldPort and ThetaBelief polymorphism."""

from __future__ import annotations

import numpy as np
import pytest

from domain.entities.rosa_roja.motif import TopologicalMotifKey
from domain.entities.rosa_roja.theta_belief import ThetaBelief
from domain.ports.rosa_roja.guided_field import GuidedFieldPort, NullGuidedField


def test_topological_motif_key_creation_and_immutability():
    motif = TopologicalMotifKey(
        direction_index=3,
        acceleration_mode=1,
        tempo_band=0,
        motif_id=42,
    )
    assert motif.direction_index == 3
    assert motif.acceleration_mode == 1
    assert motif.tempo_band == 0
    assert motif.motif_id == 42

    # Inmutable (frozen)
    with pytest.raises(Exception):
        motif.motif_id = 99


def test_theta_belief_with_topological_motifs():
    theta = ThetaBelief(alpha=0.9)
    m1 = TopologicalMotifKey(direction_index=0, acceleration_mode=1, tempo_band=0, motif_id=1)
    m2 = TopologicalMotifKey(direction_index=1, acceleration_mode=0, tempo_band=1, motif_id=2)
    m3 = TopologicalMotifKey(direction_index=2, acceleration_mode=-1, tempo_band=-1, motif_id=3)

    # 10 transiciones m1 -> m2
    for _ in range(10):
        theta.update(m1, m2)

    probs = theta.get_transition_probabilities(m1)
    assert probs[m2] == pytest.approx(1.0)
    assert theta.compute_entropy(m1) == 0.0

    # Cambio a m1 -> m3
    for _ in range(20):
        theta.update(m1, m3)

    probs_updated = theta.get_transition_probabilities(m1)
    assert probs_updated[m3] > probs_updated[m2]


def test_theta_belief_motif_serialization_roundtrip():
    theta = ThetaBelief()
    m1 = TopologicalMotifKey(direction_index=5, acceleration_mode=-2, tempo_band=1, motif_id=15)
    m2 = TopologicalMotifKey(direction_index=8, acceleration_mode=2, tempo_band=-1, motif_id=88)

    theta.update(m1, m2)
    exported = theta.export_state()

    restored = ThetaBelief()
    restored.import_state(exported)

    probs = restored.get_transition_probabilities(m1)
    assert probs[m2] == pytest.approx(1.0)
    assert m1 in restored._transitions


def test_null_guided_field_contract():
    null_field: GuidedFieldPort = NullGuidedField()
    grad, conf = null_field.get_macro_gradient(timestamp=1000.0)
    assert isinstance(grad, np.ndarray)
    assert conf == 0.0
    priors = null_field.get_motif_prior(regime="volatile")
    assert priors == {}
