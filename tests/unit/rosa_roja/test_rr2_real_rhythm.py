"""RR-2 tests: real rhythm signature with temporal dynamics."""

from __future__ import annotations

import numpy as np
import pytest

from core.orchestration.rosa_roja.domain.movement import Movement, RhythmSignature
from core.orchestration.rosa_roja.domain.trajectory import Trajectory, TerminalState
from core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator


def make_movement(delta, dt: float = 1.0, velocity: float = None, prev: Movement = None) -> Movement:
    """Helper to create Movement with real rhythm signature."""
    d = np.asarray(delta, dtype=float)
    norm = np.linalg.norm(d)
    direction = d / norm if norm > 0 else np.zeros_like(d)
    vel = velocity if velocity is not None else (norm / dt if dt > 0 else 0.0)
    rhythm = RhythmSignature(
        tempo_ratio=1.0,  # Will be overwritten by from_raw
        velocity_delta=0.0,
        acceleration=0.0,
        phase_angle=0.0,
        entropy_rate=0.1,
    )
    return Movement(
        delta_state=d,
        delta_time=dt,
        velocity=vel,
        direction=direction,
        rhythm_signature=rhythm,
        mahalanobis_distance=1.0,
        timestamp=0.0,
    )


class TestMovementRhythmSignature:
    """Tests for real tempo_ratio, acceleration, phase_angle computation."""

    def test_tempo_ratio_computed_from_prev(self):
        """Δt_i / Δt_{i-1} correctly computed."""
        m1 = Movement.from_raw(np.array([1.0, 0.0]), delta_time=2.0, timestamp=0.0)
        m2 = Movement.from_raw(np.array([2.0, 0.0]), delta_time=1.0, timestamp=1.0, prev_movement=m1)

        assert m2.rhythm_signature.tempo_ratio == pytest.approx(0.5)

    def test_tempo_ratio_unity_when_no_prev(self):
        """tempo_ratio = 1.0 when no previous movement."""
        m = Movement.from_raw(np.array([1.0, 0.0]), delta_time=2.0, timestamp=0.0)

        assert m.rhythm_signature.tempo_ratio == 1.0

    def test_acceleration_when_velocity_changes(self):
        """acceleration = (v_i - v_{i-1}) / Δt_i when prev exists."""
        m1 = Movement.from_raw(np.array([1.0, 0.0]), delta_time=1.0, timestamp=0.0)
        m2 = Movement.from_raw(np.array([4.0, 0.0]), delta_time=2.0, timestamp=1.0, prev_movement=m1)

        # v1 = 1.0/1.0 = 1.0, v2 = 4.0/2.0 = 2.0
        # acc = (2.0 - 1.0) / 2.0 = 0.5
        assert m2.rhythm_signature.acceleration == pytest.approx(0.5)

    def test_phase_angle_between_directions(self):
        """phase_angle = arccos(dot(d_i, d_{i-1}))."""
        m1 = Movement.from_raw(np.array([1.0, 0.0]), delta_time=1.0, timestamp=0.0)
        m2 = Movement.from_raw(np.array([0.0, 1.0]), delta_time=1.0, timestamp=1.0, prev_movement=m1)

        # Orthogonal directions: dot = 0, phase = π/2
        assert m2.rhythm_signature.phase_angle == pytest.approx(np.pi / 2)

    def test_same_direction_phase_zero(self):
        """phase_angle = 0 when directions are identical."""
        m1 = Movement.from_raw(np.array([1.0, 0.0]), delta_time=1.0, timestamp=0.0)
        m2 = Movement.from_raw(np.array([2.0, 0.0]), delta_time=1.0, timestamp=1.0, prev_movement=m1)

        assert m2.rhythm_signature.phase_angle == pytest.approx(0.0)

    def test_velocity_delta_correct(self):
        """velocity_delta = v_i - v_{i-1}."""
        m1 = Movement.from_raw(np.array([1.0, 0.0]), delta_time=1.0, timestamp=0.0)  # v=1
        m2 = Movement.from_raw(np.array([4.0, 0.0]), delta_time=2.0, timestamp=1.0, prev_movement=m1)  # v=2

        assert m2.rhythm_signature.velocity_delta == pytest.approx(1.0)


class TestPhiRitmoTempoSensitivity:
    """Φ_Ritmo must penalize erratic tempo and reward rhythmic pacing."""

    def test_steady_tempo_scores_higher_than_erratic(self):
        """Trajectory with constant Δt should score higher than one with variable Δt."""
        gen = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=5)

        # Build steady-tempo trajectory: all Δt = 1.0
        steady_movements = []
        for i in range(4):
            dt = 1.0
            m = Movement.from_raw(
                np.array([float(i + 1), 0.0]),
                delta_time=dt,
                timestamp=float(i),
                prev_movement=steady_movements[-1] if steady_movements else None
            )
            steady_movements.append(m)

        # Build erratic-tempo trajectory: alternating Δt = 0.5, 2.0
        erratic_movements = []
        for i in range(4):
            dt = 0.5 if i % 2 == 0 else 2.0
            m = Movement.from_raw(
                np.array([float(i + 1), 0.0]),
                delta_time=dt,
                timestamp=sum([1.0, 0.5, 2.0, 0.5][:i]),
                prev_movement=erratic_movements[-1] if erratic_movements else None
            )
            erratic_movements.append(m)

        steady_traj = Trajectory(
            movements=tuple(steady_movements),
            coherence_score=0.0,
            invalidation_step=None,
            terminal_state=TerminalState(
                state_vector=steady_movements[-1].delta_state,
                step_index=3,
                confidence=0.0,
            ),
        )
        erratic_traj = Trajectory(
            movements=tuple(erratic_movements),
            coherence_score=0.0,
            invalidation_step=None,
            terminal_state=TerminalState(
                state_vector=erratic_movements[-1].delta_state,
                step_index=3,
                confidence=0.0,
            ),
        )

        scored_steady = gen._phi_ritmo(steady_traj, lambda_t=0.1, entropy=0.1)
        scored_erratic = gen._phi_ritmo(erratic_traj, lambda_t=0.1, entropy=0.1)

        assert scored_steady.coherence_score > scored_erratic.coherence_score, (
            f"Steady tempo ({scored_steady.coherence_score:.4f}) should beat "
            f"erratic tempo ({scored_erratic.coherence_score:.4f})"
        )

    def test_steady_velocity_direction_tempo_all_constant_max_score(self):
        """When velocity, direction, and tempo all constant, ρ(T) → 1."""
        # Create perfectly consistent trajectory
        movements = []
        for i in range(4):
            m = Movement.from_raw(
                np.array([1.0, 0.0]),  # Same direction, magnitude
                delta_time=1.0,         # Same tempo
                timestamp=float(i),
                prev_movement=movements[-1] if movements else None
            )
            movements.append(m)

        traj = Trajectory(
            movements=tuple(movements),
            coherence_score=0.0,
            invalidation_step=None,
            terminal_state=TerminalState(
                state_vector=movements[-1].delta_state,
                step_index=3,
                confidence=0.0,
            ),
        )

        gen = RhythmTrajectoryGenerator()
        scored = gen._phi_ritmo(traj, lambda_t=0.0, entropy=0.0)

        # With λ=0, entropy=0: φ = ρ / |T|
        # ρ = γ * (0.4*1 + 0.4*1 + 0.2*1) = γ * 1.0 = 0.5
        # φ = 0.5 / 4 = 0.125
        assert scored.coherence_score == pytest.approx(0.125, abs=1e-4)


class TestTransitionWeightsTempo:
    """Transition weights should prefer similar tempo_ratio."""

    def test_tempo_similarity_in_weights(self):
        """Candidates with similar tempo_ratio get higher weight."""
        gen = RhythmTrajectoryGenerator()
        from_m = Movement.from_raw(np.array([1.0, 0.0]), delta_time=1.0, timestamp=0.0)  # tempo=1

        # Candidate A: same tempo (dt=1.0) -> tempo_ratio = 1.0
        cand_a = Movement.from_raw(np.array([1.0, 0.0]), delta_time=1.0, timestamp=1.0, prev_movement=from_m)
        # Candidate B: double tempo (dt=0.5) -> tempo_ratio = 0.5
        cand_b = Movement.from_raw(np.array([1.0, 0.0]), delta_time=0.5, timestamp=1.0, prev_movement=from_m)

        weights = gen._compute_transition_weights(from_m, [cand_a, cand_b], lambda_t=0.0, posterior=None)

        # A has tempo_ratio=1.0, B has 0.5
        # tempo_sim(A) = 1 - |log(1/1)| = 1.0
        # tempo_sim(B) = 1 - |log(0.5/1)| = 1 - |log(0.5)| = 1 - 0.693 = 0.307
        # Both have same vel_sim=1, dir_sim=1, so coherence weights differ only by tempo
        # Expected: weight_A > weight_B
        assert weights[0] > weights[1], f"Similar tempo should weight more: {weights}"


class TestRegression:
    """Ensure existing tests still pass."""

    def test_rr0_tests_still_pass(self):
        """Run RR-0 tests indirectly via module usage."""
        gen = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=5)
        
        # Build history
        for i in range(6):
            m = Movement.from_raw(
                np.array([float(i + 1), 0.0]),
                delta_time=1.0,
                timestamp=float(i),
                prev_movement=gen._history[-1] if gen._history else None
            )
            gen._history.append(m)
        
        latest = Movement.from_raw(
            np.array([5.0, 0.0]),
            delta_time=1.0,
            timestamp=5.0,
            prev_movement=gen._history[-1]
        )
        
        trajs = gen.generate_candidate_trajectories(latest, drift_score=0.1)
        
        assert len(trajs) > 0
        for t in trajs:
            assert 3 <= len(t.movements) <= 5
            deltas = [m.delta_state for m in t.movements]
            for a, b in zip(deltas, deltas[1:]):
                assert not np.array_equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])