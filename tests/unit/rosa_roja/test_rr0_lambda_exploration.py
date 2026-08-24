"""RR-0 tests: λ-driven exploration in the random walk + clean truncation (no padding)."""

from __future__ import annotations

import random

import numpy as np
import pytest

from core.orchestration.rosa_roja.domain.movement import Movement, RhythmSignature
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


def branchy_generator() -> tuple[RhythmTrajectoryGenerator, Movement]:
    """Graph where the start state has 4 successors: 1 coherent, 3 orthogonal."""
    gen = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=5, top_k=2)
    start = make_movement([1.0, 0.0])
    successors = [
        make_movement([1.0, 0.0]),   # coherent: same velocity, same direction
        make_movement([0.0, 1.0]),   # orthogonal
        make_movement([-1.0, 0.0]),  # orthogonal
        make_movement([0.0, -1.0]),  # orthogonal
    ]
    gen.set_transition_graph({tuple(np.round(start.delta_state, 2)): successors})
    return gen, start


def first_step_distribution(gen, start, lambda_t: float, n: int = 800) -> dict:
    counts: dict = {}
    for i in range(n):
        random.seed(i)
        np.random.seed(i)
        traj = gen._random_walk(start, lambda_t)
        key = tuple(np.round(traj.movements[1].delta_state, 2))
        counts[key] = counts.get(key, 0) + 1
    return counts


def shannon_entropy(counts: dict) -> float:
    total = sum(counts.values())
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log(probs)))


class TestLambdaModulatesExploration:
    """λ_t must change the generation process, not only the scoring."""

    def test_low_lambda_concentrates_on_coherent_successor(self):
        gen, start = branchy_generator()
        coherent_key = (1.0, 0.0)

        counts = first_step_distribution(gen, start, lambda_t=0.05)

        assert counts[coherent_key] / sum(counts.values()) > 0.35

    def test_high_lambda_flattens_distribution(self):
        gen, start = branchy_generator()

        counts = first_step_distribution(gen, start, lambda_t=0.95)

        total = sum(counts.values())
        coherent_freq = counts.get((1.0, 0.0), 0) / total
        assert coherent_freq < 0.32

    def test_entropy_of_sampling_grows_with_lambda(self):
        gen, start = branchy_generator()

        h_low = shannon_entropy(first_step_distribution(gen, start, lambda_t=0.05))
        h_high = shannon_entropy(first_step_distribution(gen, start, lambda_t=0.95))

        assert h_high > h_low

    def test_extreme_lambdas_bound_weights(self):
        gen, start = branchy_generator()
        successors = gen._transition_graph[(1.0, 0.0)]

        w_exploit = gen._compute_transition_weights(start, successors, lambda_t=0.0)
        w_explore = gen._compute_transition_weights(start, successors, lambda_t=1.0)

        assert np.allclose(w_explore, np.full(len(successors), 1.0 / len(successors)))
        assert w_exploit.max() == pytest.approx(w_exploit.sum(), rel=1e-6) or w_exploit.max() > w_exploit.min()
        assert np.isclose(w_exploit.sum(), 1.0)
        assert np.isclose(w_explore.sum(), 1.0)


class TestCleanTruncation:
    """The walk must stop cleanly instead of padding with repeated movements."""

    def test_dead_end_stops_without_padding(self):
        gen = RhythmTrajectoryGenerator(min_trajectory_len=11, max_trajectory_len=15)
        start = make_movement([1.0, 0.0])
        gen._transition_graph[(1.0, 0.0)] = [make_movement([2.0, 0.0], velocity=2.0)]

        traj = gen._random_walk(start, lambda_t=0.3)

        assert traj.metadata["truncated"] is True
        assert traj.metadata["stop_reason"] == "dead_end"
        assert len(traj.movements) == 2

    def test_no_consecutive_duplicate_movements(self):
        gen = RhythmTrajectoryGenerator(min_trajectory_len=11, max_trajectory_len=15)
        start = make_movement([1.0, 0.0])
        gen._transition_graph[(1.0, 0.0)] = [make_movement([2.0, 0.0], velocity=2.0)]

        traj = gen._random_walk(start, lambda_t=0.3)

        deltas = [m.delta_state for m in traj.movements]
        for a, b in zip(deltas, deltas[1:]):
            assert not np.array_equal(a, b)

    def test_cycle_stop_reports_reason(self):
        gen = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=15)
        a = make_movement([1.0, 0.0])
        b = make_movement([2.0, 0.0], velocity=2.0)
        gen._transition_graph[tuple(np.round(a.delta_state, 2))] = [b]
        gen._transition_graph[tuple(np.round(b.delta_state, 2))] = [
            make_movement([1.0, 0.0])
        ]

        traj = gen._random_walk(a, lambda_t=0.0)

        assert traj.metadata["stop_reason"] == "cycle"

    def test_max_length_stop_is_not_truncated(self):
        gen = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=4)
        chain = [make_movement([float(i), 0.0]) for i in range(1, 6)]
        for prev, nxt in zip(chain, chain[1:]):
            gen._transition_graph.setdefault(
                tuple(np.round(prev.delta_state, 2)), []
            ).append(nxt)

        traj = gen._random_walk(chain[0], lambda_t=0.0)

        assert len(traj.movements) == 4
        assert traj.metadata["stop_reason"] == "max_length"
        assert traj.metadata["truncated"] is False


class TestGenerationPathIntegrity:
    """Full generation path: trajectories respect bounds and never pad."""

    def test_generated_trajectories_have_no_duplicates(self):
        gen = RhythmTrajectoryGenerator(
            min_trajectory_len=3, max_trajectory_len=5, top_k=2, oversample_factor=2
        )
        # Interleave main-chain and branch observations so the rebuilt
        # transition graph contains branching states.
        for i in range(5):
            gen._history.append(make_movement([float(i + 1), 0.0]))
            gen._history.append(make_movement([float(i + 2), 1.0]))

        latest = make_movement([float(4), 0.0])
        trajs = gen.generate_candidate_trajectories(latest, drift_score=0.1)

        assert trajs
        for t in trajs:
            assert 3 <= len(t.movements) <= 5
            assert "truncated" in t.metadata
            assert "stop_reason" in t.metadata
            deltas = [m.delta_state for m in t.movements]
            for a, b in zip(deltas, deltas[1:]):
                assert not np.array_equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
