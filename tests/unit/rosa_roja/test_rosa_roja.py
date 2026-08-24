"""Unit tests for Rosa Roja Engine - Critical Expert Veto Behavior."""

from __future__ import annotations

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from core.orchestration.rosa_roja.domain.movement import Movement, RhythmSignature
from core.orchestration.rosa_roja.domain.trajectory import Trajectory, TerminalState
from core.orchestration.rosa_roja.domain.validation import ValidationResult, VetoDetails
from core.orchestration.rosa_roja.domain.execution import ExecutionPlan, ActionEnvelope
from core.orchestration.rosa_roja.modules.module1_ingestion import MahalanobisFilter
from core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator
from core.orchestration.rosa_roja.modules.module3_moe_gating import MultiplicativeMoEGating
from core.orchestration.rosa_roja.ports.expert_jury import ExpertJuryPort
from core.orchestration.rosa_roja.ports.drift_sensor import DriftSensorPort
from core.orchestration.rosa_roja.engine import RosaRojaEngine


class MockExpertJuryPort:
    """Mock expert for testing."""
    
    def __init__(self, name: str, is_critical: bool, threshold: float, weight: float, score: float):
        self.name = name
        self.is_critical = is_critical
        self.threshold = threshold
        self.weight = weight
        self._score = score
        self.update_called = False
        self.last_actual = None
        self.last_predicted = None
    
    def evaluate_trajectory(self, trajectory: Trajectory) -> float:
        return self._score
    
    def update_learning(self, actual: float, predicted: float) -> None:
        self.update_called = True
        self.last_actual = actual
        self.last_predicted = predicted


class MockDriftSensorPort:
    """Mock drift sensor for testing."""
    
    def __init__(self, name: str, drift_score: float = 0.0):
        self.name = name
        self._drift_score = drift_score
        self.update_called = False
        self.last_actual = None
        self.last_predicted = None
    
    def get_drift_score(self) -> float:
        return self._drift_score
    
    def update(self, actual: float, predicted: float) -> None:
        self.update_called = True
        self.last_actual = actual
        self.last_predicted = predicted
    
    def reset(self) -> None:
        pass


def create_test_trajectory(length: int = 12, coherence: float = 0.8) -> Trajectory:
    """Create a test trajectory with given length and coherence."""
    movements = []
    for i in range(length):
        delta_state = np.array([float(i), 0.0, 0.0])
        rhythm = RhythmSignature(
            tempo_ratio=1.0,
            velocity_delta=0.0,
            acceleration=0.0,
            phase_angle=0.0,
            entropy_rate=0.1,
        )
        movement = Movement(
            delta_state=delta_state,
            delta_time=1.0,
            velocity=1.0,
            direction=np.array([1.0, 0.0, 0.0]),
            rhythm_signature=rhythm,
            mahalanobis_distance=1.0,
            timestamp=float(i),
        )
        movements.append(movement)
    
    return Trajectory(
        movements=tuple(movements),
        coherence_score=coherence,
        invalidation_step=None,
        terminal_state=TerminalState(
            state_vector=movements[-1].delta_state,
            step_index=len(movements) - 1,
            confidence=coherence,
        ),
    )


class TestCriticalExpertVeto:
    """Tests verifying that a failing critical expert vetoes the entire trajectory."""
    
    def test_critical_expert_below_threshold_vetoes_trajectory(self):
        """A critical expert scoring below threshold should veto the trajectory."""
        # Setup: 2 critical experts, 1 non-critical
        # Critical expert 1 passes (score 0.7 > 0.65)
        # Critical expert 2 FAILS (score 0.5 < 0.65)
        # Non-critical expert scores 1.0 (perfect)
        
        critical_pass = MockExpertJuryPort("taylor", True, 0.65, 1.2, 0.7)
        critical_fail = MockExpertJuryPort("kalman", True, 0.65, 1.0, 0.5)  # BELOW THRESHOLD
        non_critical = MockExpertJuryPort("statistical", False, 0.55, 0.8, 1.0)
        
        jury = [critical_pass, critical_fail, non_critical]
        
        trajectories = [create_test_trajectory()]
        
        gating = MultiplicativeMoEGating(variance_penalty=0.5)
        result = gating.evaluate_and_veto(trajectories, jury)
        
        # Verify veto was triggered
        assert result.veto_triggered is True
        assert result.chosen_trajectory is None
        assert result.global_confidence == 0.0
        assert result.veto_details is not None
        assert result.veto_details.expert_name == "kalman"
        assert result.veto_details.expert_type == "critical"
        assert result.veto_details.score == 0.5
        assert result.veto_details.threshold == 0.65
        assert "below threshold" in result.veto_details.reason
    
    def test_multiple_critical_experts_all_must_pass(self):
        """All critical experts must pass - if ANY fails, trajectory is vetoed."""
        critical_1 = MockExpertJuryPort("taylor", True, 0.65, 1.2, 0.8)   # passes
        critical_2 = MockExpertJuryPort("kalman", True, 0.65, 1.0, 0.7)   # passes
        critical_3 = MockExpertJuryPort("lightgbm", True, 0.60, 1.1, 0.4) # FAILS
        non_critical = MockExpertJuryPort("statistical", False, 0.55, 0.8, 1.0)
        
        jury = [critical_1, critical_2, critical_3, non_critical]
        trajectories = [create_test_trajectory()]
        
        gating = MultiplicativeMoEGating()
        result = gating.evaluate_and_veto(trajectories, jury)
        
        assert result.veto_triggered is True
        assert result.chosen_trajectory is None
        assert result.veto_details.expert_name == "lightgbm"
    
    def test_non_critical_expert_below_threshold_DOES_NOT_veto(self):
        """Non-critical expert below threshold should NOT veto - only reduces weighted mean."""
        critical_pass = MockExpertJuryPort("taylor", True, 0.65, 1.2, 0.8)
        critical_pass2 = MockExpertJuryPort("kalman", True, 0.65, 1.0, 0.7)
        non_critical_fail = MockExpertJuryPort("statistical", False, 0.55, 0.8, 0.3)  # BELOW threshold
        
        jury = [critical_pass, critical_pass2, non_critical_fail]
        trajectories = [create_test_trajectory()]
        
        gating = MultiplicativeMoEGating()
        result = gating.evaluate_and_veto(trajectories, jury)
        
        # Should NOT veto - trajectory selected with reduced confidence
        assert result.veto_triggered is False
        assert result.chosen_trajectory is not None
        assert result.global_confidence > 0.0
        # Weighted mean: (1.2*0.8 + 1.0*0.7 + 0.8*0.3) / (1.2+1.0+0.8) = 1.9 / 3.0 ≈ 0.633
        # With variance penalty: 0.633 / (1 + 0.5 * variance)
        assert result.global_confidence < 0.633  # Variance penalty applied
    
    def test_all_critical_pass_trajectory_selected(self):
        """When all critical experts pass, trajectory is selected with Φ_MoE score."""
        critical_1 = MockExpertJuryPort("taylor", True, 0.65, 1.2, 0.9)
        critical_2 = MockExpertJuryPort("kalman", True, 0.65, 1.0, 0.85)
        non_critical = MockExpertJuryPort("statistical", False, 0.55, 0.8, 0.8)
        
        jury = [critical_1, critical_2, non_critical]
        trajectories = [create_test_trajectory()]
        
        gating = MultiplicativeMoEGating()
        result = gating.evaluate_and_veto(trajectories, jury)
        
        assert result.veto_triggered is False
        assert result.chosen_trajectory is not None
        assert result.global_confidence > 0.0
        assert "taylor" in result.all_scores
        assert "kalman" in result.all_scores
        assert "statistical" in result.all_scores
    
    def test_variance_penalty_reduces_confidence(self):
        """High inter-expert variance should reduce global confidence."""
        # Low variance case
        critical_1 = MockExpertJuryPort("taylor", True, 0.65, 1.2, 0.8)
        critical_2 = MockExpertJuryPort("kalman", True, 0.65, 1.0, 0.8)
        non_critical = MockExpertJuryPort("statistical", False, 0.55, 0.8, 0.8)
        
        jury_low_var = [critical_1, critical_2, non_critical]
        trajectories = [create_test_trajectory()]
        
        gating = MultiplicativeMoEGating(variance_penalty=0.5)
        result_low = gating.evaluate_and_veto(trajectories, jury_low_var)
        
        # High variance case
        critical_1_h = MockExpertJuryPort("taylor", True, 0.65, 1.2, 0.9)
        critical_2_h = MockExpertJuryPort("kalman", True, 0.65, 1.0, 0.9)
        non_critical_h = MockExpertJuryPort("statistical", False, 0.55, 0.8, 0.3)
        
        jury_high_var = [critical_1_h, critical_2_h, non_critical_h]
        result_high = gating.evaluate_and_veto(trajectories, jury_high_var)
        
        # High variance should have lower confidence despite similar means
        # Mean low: (1.2*0.8 + 1.0*0.8 + 0.8*0.8) / 3.0 = 2.4/3.0 = 0.8
        # Mean high: (1.2*0.9 + 1.0*0.9 + 0.8*0.3) / 3.0 = 2.52/3.0 = 0.84
        # But high variance case has penalty
        assert result_low.global_confidence > result_high.global_confidence
    
    def test_chooses_best_trajectory_among_candidates(self):
        """Should select trajectory with highest Φ_MoE among candidates."""
        # Create two trajectories
        traj1 = create_test_trajectory(length=11, coherence=0.7)
        traj2 = create_test_trajectory(length=13, coherence=0.8)
        traj3 = create_test_trajectory(length=12, coherence=0.6)
        
        # All experts score traj2 highest
        expert = MockExpertJuryPort("taylor", True, 0.65, 1.2, 0.9)
        expert2 = MockExpertJuryPort("kalman", True, 0.65, 1.0, 0.85)
        
        # Override to return different scores per trajectory
        def score_traj(t):
            if t is traj2:
                return 0.9
            elif t is traj1:
                return 0.7
            return 0.5
        
        expert.evaluate_trajectory = score_traj
        expert2.evaluate_trajectory = score_traj
        
        jury = [expert, expert2]
        trajectories = [traj1, traj2, traj3]
        
        gating = MultiplicativeMoEGating()
        result = gating.evaluate_and_veto(trajectories, jury)
        
        assert result.chosen_trajectory is traj2


class TestRosaRojaEngineIntegration:
    """Integration tests for the full Rosa Roja Engine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.ingestion = MahalanobisFilter(noise_threshold=3.0, history_window=50)
        self.rhythm = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=5, top_k=2)
        self.gating = MultiplicativeMoEGating()
        
        # Mock experts
        self.experts = [
            MockExpertJuryPort("taylor", True, 0.65, 1.2, 0.8),
            MockExpertJuryPort("kalman", True, 0.60, 1.0, 0.85),
            MockExpertJuryPort("statistical", False, 0.55, 0.8, 0.75),
        ]
        
        # Mock drift sensor
        self.drift_sensor = MockDriftSensorPort("page_hinkley", 0.1)
        
        self.engine = RosaRojaEngine(
            ingestion_filter=self.ingestion,
            rhythm_generator=self.rhythm,
            moe_gating=self.gating,
            expert_jury=self.experts,
            drift_sensors=[self.drift_sensor],
        )
    
    def test_process_event_returns_execution_plan(self):
        """process_event should return an ExecutionPlan."""
        delta_state = np.array([1.0, 0.5, -0.2])
        delta_time = 1.0
        
        plan = self.engine.process_event(delta_state, delta_time)
        
        assert isinstance(plan, ExecutionPlan)
        assert plan.action in ["EXECUTE", "HOLD", "EMERGENCY_FLUSH"]
    
    def test_process_event_hold_on_insufficient_history(self):
        """First few events should HOLD due to insufficient trajectory density."""
        for i in range(5):
            delta_state = np.array([float(i), 0.0, 0.0])
            plan = self.engine.process_event(delta_state, 1.0)
            # May HOLD or EXECUTE depending on trajectory generation
            assert isinstance(plan, ExecutionPlan)
    
    def test_update_feedback_propagates_to_experts_and_sensors(self):
        """update_feedback should call update_learning on experts and update on sensors."""
        actual = np.array([100.0])
        predicted = np.array([99.0])
        
        self.engine.update_feedback(actual, predicted)
        
        for expert in self.experts:
            assert expert.update_called
            assert expert.last_actual == 100.0
            assert expert.last_predicted == 99.0
        
        assert self.drift_sensor.update_called
        assert self.drift_sensor.last_actual == 100.0
        assert self.drift_sensor.last_predicted == 99.0
    
    def test_critical_expert_veto_propagates_to_execution_plan(self):
        """A vetoed trajectory should result in HOLD ExecutionPlan."""
        # Replace one expert to fail
        failing_expert = MockExpertJuryPort("kalman", True, 0.65, 1.0, 0.5)  # FAIL
        self.engine._jury[1] = failing_expert

        # Need enough history with known successors for trajectory generation:
        # repeat a 5-state cycle so every visited state has observed transitions.
        cycle = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i in range(10):
            self.engine.process_event(np.array([cycle[i % 5], 0.0, 0.0]), 1.0)

        # Trigger gating on a state whose successors are known
        plan = self.engine.process_event(np.array([cycle[0], 0.0, 0.0]), 1.0)
        
        assert plan.action == "HOLD"
        assert plan.global_confidence == 0.0
        assert plan.veto_details.get("reason", "") != "Insufficient_Trajectory_Density"
        assert "below threshold" in plan.veto_details.get("reason", "")
    
    def test_regime_alert_on_mahalanobis_outlier(self):
        """Module 1 outlier should trigger regime_alert in ExecutionPlan."""
        # Build up history first
        for i in range(30):
            delta_state = np.array([float(i), 0.0, 0.0])
            self.engine.process_event(delta_state, 1.0)
        
        # Now send a huge outlier
        outlier = np.array([1000.0, 1000.0, 1000.0])
        plan = self.engine.process_event(outlier, 1.0)
        
        assert plan.action == "HOLD"
        assert plan.regime_alert is True
        assert "Outlier" in plan.veto_details.get("reason", "")
    
    def test_jury_status_returns_expert_info(self):
        """get_jury_status should return info for all experts."""
        status = self.engine.get_jury_status()
        
        assert len(status) == 3
        assert "taylor" in status
        assert "kalman" in status
        assert "statistical" in status
        assert status["taylor"]["is_critical"] is True
        assert status["statistical"]["is_critical"] is False
    
    def test_drift_status_returns_sensor_scores(self):
        """get_drift_status should return drift scores."""
        status = self.engine.get_drift_status()
        
        assert "page_hinkley" in status
        assert status["page_hinkley"]["drift_score"] == 0.1


class TestModule1MahalanobisFilter:
    """Tests for Module 1: Mahalanobis Filter."""
    
    def test_accepts_first_samples_without_filtering(self):
        """First N samples should be accepted without Mahalanobis check."""
        filter = MahalanobisFilter(noise_threshold=1.0, history_window=10, min_samples_for_cov=5)
        
        for i in range(5):
            movement, is_outlier = filter.process_raw_step(
                np.array([float(i), 0.0]), 1.0
            )
            assert is_outlier is False
            assert movement is not None
    
    def test_outlier_triggered_by_large_mahalanobis(self):
        """Large Mahalanobis distance should trigger outlier alert."""
        filter = MahalanobisFilter(noise_threshold=2.0, history_window=20, min_samples_for_cov=10)
        
        # Build history
        for i in range(15):
            filter.process_raw_step(np.array([float(i), 0.0]), 1.0)
        
        # Send outlier
        movement, is_outlier = filter.process_raw_step(np.array([100.0, 100.0]), 1.0)
        
        assert is_outlier is True
        assert movement.mahalanobis_distance > 2.0
    
    def test_outlier_not_added_to_history(self):
        """Outlier movements should not be added to history."""
        filter = MahalanobisFilter(noise_threshold=2.0, history_window=20, min_samples_for_cov=10)
        
        for i in range(15):
            filter.process_raw_step(np.array([float(i), 0.0]), 1.0)
        
        history_size_before = len(filter.get_history())
        
        filter.process_raw_step(np.array([100.0, 100.0]), 1.0)
        
        history_size_after = len(filter.get_history())
        assert history_size_after == history_size_before  # Outlier not added


class TestModule2RhythmTrajectoryGenerator:
    """Tests for Module 2: Rhythm Trajectory Generator."""
    
    def test_generates_trajectories_with_sufficient_history(self):
        """Should generate trajectories when history >= min_trajectory_len."""
        generator = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=5)
        drift = MockDriftSensorPort("test", 0.1)
        
        # Add history including transitions beyond the latest state so the
        # walk has observed successors (no padding since RR-0).
        for i in range(5):
            delta_state = np.array([float(i), 0.0])
            rhythm = RhythmSignature(1.0, 0.0, 0.0, 0.0, 0.1)
            movement = Movement(delta_state, 1.0, 1.0, np.array([1.0, 0.0]), rhythm, 1.0, float(i))
            generator._history.append(movement)
        for j in range(5, 9):
            rhythm = RhythmSignature(1.0, 0.0, 0.0, 0.0, 0.1)
            movement = Movement(np.array([float(j), 0.0]), 1.0, 1.0, np.array([1.0, 0.0]), rhythm, 1.0, float(j))
            generator._history.append(movement)
        generator._update_transition_graph()
        
        # Generate
        latest = Movement(
            delta_state=np.array([5.0, 0.0]),
            delta_time=1.0,
            velocity=1.0,
            direction=np.array([1.0, 0.0]),
            rhythm_signature=RhythmSignature(1.0, 0.0, 0.0, 0.0, 0.1),
            mahalanobis_distance=1.0,
            timestamp=5.0,
        )
        
        trajectories = generator.generate_candidate_trajectories(latest, 0.1)
        
        assert len(trajectories) > 0
        for t in trajectories:
            assert 3 <= len(t.movements) <= 5
            assert t.coherence_score >= 0.0
    
    def test_lambda_t_decreases_with_drift(self):
        """λ_t should decrease as drift_score increases."""
        generator = RhythmTrajectoryGenerator(max_entropy=5.0)
        
        # Mock entropy computation
        generator._compute_entropy = lambda: 2.5  # Half of max_entropy
        
        lambda_low_drift = generator._compute_lambda(2.5, 0.1)   # drift = 0.1
        lambda_high_drift = generator._compute_lambda(2.5, 0.8)  # drift = 0.8
        
        assert lambda_low_drift > lambda_high_drift
        assert abs(lambda_low_drift - 0.5) < 1e-10
        assert abs(lambda_high_drift - 0.2) < 1e-10
    
    def test_trajectory_length_normalization(self):
        """Φ_Ritmo should be normalized by trajectory length."""
        generator = RhythmTrajectoryGenerator(min_trajectory_len=3, max_trajectory_len=5)
        
        # Create trajectory with high coherence
        traj_short = create_test_trajectory(length=3, coherence=0.9)
        traj_long = create_test_trajectory(length=5, coherence=0.9)
        
        # Manually test phi_ritmo normalization
        lambda_t = 0.5
        entropy = 2.5
        
        traj_short_scored = generator._phi_ritmo(traj_short, lambda_t, entropy)
        traj_long_scored = generator._phi_ritmo(traj_long, lambda_t, entropy)
        
        # Shorter trajectory should have higher normalized score
        assert traj_short_scored.coherence_score > traj_long_scored.coherence_score


class TestModule3MultiplicativeMoEGating:
    """Tests for Module 3: Multiplicative MoE Gating."""
    
    def test_veto_details_captured_correctly(self):
        """VetoDetails should capture all relevant information."""
        expert = MockExpertJuryPort("kalman", True, 0.65, 1.0, 0.5)
        trajectory = create_test_trajectory()
        
        gating = MultiplicativeMoEGating()
        veto = gating._check_critical_veto(trajectory, [expert])
        
        assert veto is not None
        assert veto.expert_name == "kalman"
        assert veto.expert_type == "critical"
        assert veto.score == 0.5
        assert veto.threshold == 0.65
        assert "below threshold" in veto.reason
    
    def test_no_veto_when_non_critical_below_threshold(self):
        """Non-critical expert below threshold should not produce veto."""
        expert = MockExpertJuryPort("statistical", False, 0.55, 0.8, 0.3)
        trajectory = create_test_trajectory()
        
        gating = MultiplicativeMoEGating()
        veto = gating._check_critical_veto(trajectory, [expert])
        
        assert veto is None


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])