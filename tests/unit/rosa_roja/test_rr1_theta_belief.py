"""RR-1 tests: explicit Theta posterior (ThetaBelief) + integration in Module 2."""

from __future__ import annotations

import numpy as np
import pytest

from core.orchestration.rosa_roja.domain.movement import Movement, RhythmSignature
from core.orchestration.rosa_roja.domain.theta_belief import ThetaBelief
from core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator


def make_movement(delta, velocity: float = 1.0) -> Movement:
    d = np.asarray(delta, dtype=float)
    norm = np.linalg.norm(d)
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
        velocity=velocity,
        direction=direction,
        rhythm_signature=rhythm,
        mahalanobis_distance=1.0,
        timestamp=0.0,
    )


class TestDecayAndRegimeMigration:
    """Probability mass must migrate from old transitions to new ones."""

    def test_old_transition_decays_when_new_regime_observed(self):
        theta = ThetaBelief(alpha=0.95)
        a, b, c = (1.0,), (2.0,), (3.0,)

        for _ in range(10):
            theta.update(a, b)
        assert theta.get_transition_probabilities(a)[b] == pytest.approx(1.0)

        # New regime: A -> C becomes the observed pattern
        for _ in range(40):
            theta.update(a, c)

        probs = theta.get_transition_probabilities(a)
        assert probs[c] > 0.9
        assert probs.get(b, 0.0) < 0.1
        assert probs[c] > probs[b]

    def test_stale_transitions_are_pruned(self):
        theta = ThetaBelief(alpha=0.5, min_weight_threshold=1e-2)
        a, b, c = (1.0,), (2.0,), (3.0,)

        theta.update(a, b)
        for _ in range(30):
            theta.update(a, c)

        # b's weight decayed below pruning threshold
        assert b not in theta._transitions[a]
        assert c in theta._transitions[a]


class TestPosteriorEntropy:
    """H(Θ|D_t) must be an exact normalized entropy in [0, 1]."""

    def test_deterministic_transition_has_zero_entropy(self):
        theta = ThetaBelief()
        a, b = (1.0,), (2.0,)
        for _ in range(5):
            theta.update(a, b)

        assert theta.compute_entropy(a) == 0.0

    def test_uniform_branches_have_maximal_entropy(self):
        theta = ThetaBelief()
        a, b, c = (1.0,), (2.0,), (3.0,)
        theta._transitions[a] = {b: 1.0, c: 1.0}

        assert theta.compute_entropy(a) == pytest.approx(1.0)

    def test_entropy_bounds_hold_for_skewed_distribution(self):
        theta = ThetaBelief()
        a, b, c = (1.0,), (2.0,), (3.0,)
        theta._transitions[a] = {b: 9.0, c: 1.0}

        h = theta.compute_entropy(a)
        assert 0.0 < h < 1.0

    def test_empty_belief_has_maximum_uncertainty(self):
        theta = ThetaBelief()

        assert theta.compute_entropy() == 1.0
        assert theta.compute_entropy((7.0,)) == 1.0

    def test_unknown_current_state_falls_back_to_global_average(self):
        theta = ThetaBelief()
        a, b, c, d = (1.0,), (2.0,), (3.0,), (4.0,)
        theta._transitions[a] = {b: 1.0}
        theta._transitions[c] = {d: 1.0}

        # Both known states are deterministic -> global average is 0.0
        assert theta.compute_entropy() == 0.0


class TestReset:
    def test_reset_clears_posterior(self):
        theta = ThetaBelief()
        theta.update((1.0,), (2.0,))
        theta.reset()

        assert theta.total_updates == 0
        assert theta.get_transition_probabilities((1.0,)) == {}
        assert theta.compute_entropy() == 1.0


class TestModule2Integration:
    """RhythmTrajectoryGenerator must update and consume the belief natively."""

    def test_generator_updates_belief_with_each_movement(self):
        gen = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=5)

        for i in range(6):
            latest = make_movement([float(i + 1), 0.0])
            gen.generate_candidate_trajectories(latest, drift_score=0.0)

        # ThetaBelief is now managed by ThetaBeliefManager
        assert gen._theta_manager.theta.total_updates >= 5

    def test_generator_delegates_entropy_to_belief(self):
        gen = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=5)

        # Deterministic chain: every state has exactly one observed successor.
        for i in range(8):
            latest = make_movement([float(i + 1), 0.0])
            gen.generate_candidate_trajectories(latest, drift_score=0.0)

        current_key = tuple(np.round(np.array([8.0, 0.0]), 2))
        assert gen._latest_state_key == current_key
        # Entropy is now computed by ThetaBeliefManager
        assert gen._theta_manager.compute_entropy(current_key) == 0.0

    def test_belief_weights_drive_walk_sampling(self):
        gen = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=5)
        start_key = (1.0, 0.0)
        favored = (2.0, 0.0)
        rival = (0.0, 2.0)

        # Belief strongly favors start->favored over start->rival
        gen._theta_manager.theta.alpha = 1.0  # no decay for exact counts
        for _ in range(20):
            gen._theta_manager.theta.update(start_key, favored)
        for _ in range(2):
            gen._theta_manager.theta.update(start_key, rival)

        gen._transition_graph[start_key] = [
            make_movement(list(favored)),
            make_movement(list(rival)),
        ]
        gen._latest_state_key = start_key

        counts = {favored: 0, rival: 0}
        import random

        for i in range(200):
            random.seed(i)
            np.random.seed(i)
            traj = gen._walk_sampler._random_walk(make_movement([1.0, 0.0]), lambda_t=0.0)
            key = tuple(np.round(traj.movements[1].delta_state, 2))
            if key in counts:
                counts[key] += 1

        total = counts[favored] + counts[rival]
        assert total > 100
        assert counts[favored] / total > 0.8

    def test_reset_clears_belief(self):
        gen = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=5)
        for i in range(5):
            gen.generate_candidate_trajectories(make_movement([float(i), 0.0]), drift_score=0.0)

        gen.reset()

        assert gen._theta_manager.theta.total_updates == 0
        assert gen._latest_state_key is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
