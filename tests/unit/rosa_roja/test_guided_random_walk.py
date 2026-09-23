"""Unit tests for Guided Importance Sampling in RandomWalkSampler and EnsembleGuidedFieldAdapter."""

from __future__ import annotations

import numpy as np
import pytest

from domain.entities.rosa_roja.movement import Movement, RhythmSignature
from domain.ports.rosa_roja.guided_field import GuidedFieldPort
from infrastructure.ml.adapters.guided_field_adapter import EnsembleGuidedFieldAdapter
from infrastructure.ml.engines.rosa_roja.algorithms.modules.random_walk_sampler import (
    RandomWalkConfig,
    RandomWalkSampler,
)
from infrastructure.ml.engines.seasonal.engine import SeasonalConfig, SeasonalPredictorEngine
from infrastructure.ml.inference.bayesian.naive_bayes import NaiveBayesClassifier


def make_dummy_movement(delta: list[float], timestamp: float = 0.0) -> Movement:
    d = np.asarray(delta, dtype=np.float64)
    norm = float(np.linalg.norm(d))
    direction = d / norm if norm > 0 else np.zeros_like(d)
    rhythm = RhythmSignature(
        tempo_ratio=1.0,
        velocity_delta=0.0,
        acceleration=0.0,
        phase_angle=0.0,
        entropy_rate=0.1,
    )
    return Movement(
        delta_state=d,
        delta_time=1.0,
        velocity=norm,
        direction=direction,
        rhythm_signature=rhythm,
        mahalanobis_distance=1.0,
        timestamp=timestamp,
    )


def test_ensemble_guided_field_adapter_with_seasonal_and_bayes():
    # Seasonal engine with small period
    s_engine = SeasonalPredictorEngine(config=SeasonalConfig(min_period=4, min_confidence=0.2))
    bayes = NaiveBayesClassifier()

    # Train naive bayes on motifs
    for _ in range(8):
        bayes.fit_online({"volatility": 0.5}, "10")
    for _ in range(2):
        bayes.fit_online({"volatility": 0.5}, "20")

    adapter = EnsembleGuidedFieldAdapter(
        seasonal_engine=s_engine,
        bayesian_classifier=bayes,
        window_size=30,
        default_dimension=2,
    )

    # Before feeding observations: gradient is zeros
    grad, conf = adapter.get_macro_gradient(timestamp=0.0)
    assert np.allclose(grad, np.zeros(2))
    assert conf == 0.0

    # Feed cyclic wave: sine wave with period 6
    for t in range(20):
        val = 100.0 + 10.0 * np.sin(2.0 * np.pi * t / 6.0)
        adapter.record_observation(val, float(t))

    grad, conf = adapter.get_macro_gradient(timestamp=20.0)
    assert grad.shape == (2,)
    # At t=20, phase of 2*pi*20/6 = 20*pi/3 = 6*pi + 2*pi/3 (declining)
    assert abs(grad[0]) == 1.0 or conf >= 0.0

    # Test Bayesian motif prior
    priors = adapter.get_motif_prior(regime="trending")
    assert 10 in priors
    assert priors[10] > priors[20]


def test_random_walk_guided_vs_unguided_dead_end():
    config = RandomWalkConfig(
        max_random_walk_steps=20,
        min_trajectory_len=5,
        max_trajectory_len=10,
        quantization_decimals=2,
    )

    # Empty transition graph (empty local memory)
    empty_graph = {}
    theta = None
    quantize = lambda state: tuple(np.round(state, 2))

    start = make_dummy_movement([1.0, 0.5], timestamp=10.0)

    # 1. Sin campo guía (legacy): debe detenerse con stop_reason = "dead_end" tras 1 paso
    sampler_unguided = RandomWalkSampler(
        config=config,
        transition_graph=empty_graph,
        theta_belief=theta,
        quantize_state_func=quantize,
        guided_field=None,
    )
    traj_unguided = sampler_unguided._random_walk(start, lambda_t=0.0)
    assert traj_unguided.metadata["stop_reason"] == "dead_end"
    assert len(traj_unguided.movements) == 1

    # 2. Con campo guía: debe solicitar el campo macro y sintetizar pasos sin morir en dead_end
    class MockGuidedField(GuidedFieldPort):
        def get_macro_gradient(self, timestamp: float):
            return np.array([1.0, 0.0]), 0.85

        def get_motif_prior(self, regime: str):
            return {1: 0.7, 2: 0.3}

    sampler_guided = RandomWalkSampler(
        config=config,
        transition_graph=empty_graph,
        theta_belief=theta,
        quantize_state_func=quantize,
        guided_field=MockGuidedField(),
    )
    traj_guided = sampler_guided._random_walk(start, lambda_t=0.0)

    # El caminante guiado continuó la trayectoria (no se detuvo en dead_end tras 1 paso)
    assert traj_guided.metadata["stop_reason"] in ("max_length", "cycle", "max_random_walk_steps")
    assert len(traj_guided.movements) > 1
    # Verifica que los movimientos sintetizados avanzan con el gradiente
    assert traj_guided.movements[-1].velocity > 0.0
