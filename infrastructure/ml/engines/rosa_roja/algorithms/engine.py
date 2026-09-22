"""Rosa Roja Engine: Master System Orchestrator (Phase-Space Continuous Dynamics)."""

from __future__ import annotations

import logging
import hashlib
import struct
from collections.abc import Sequence
from typing import Any
import numpy as np

from .domain.engine_persistence import RosaRojaPersistenceMixin
from .domain.execution import ActionEnvelope, ExecutionPlan
from .domain.state_machine import StateMachine
from .domain.trajectory_tracker import TrajectoryTracker
from .domain.validation import ValidationResult
from .modules.module1_ingestion import MahalanobisFilter
from .modules.module3_moe_gating import MultiplicativeMoEGating
from .modules.rhythm_generator import RhythmTrajectoryGenerator
from .ports.drift_sensor import DriftSensorPort
from .ports.expert_jury import ExpertJuryPort
from .ports.state_store import MLStateStore

logger = logging.getLogger(__name__)


class RosaRojaEngine(RosaRojaPersistenceMixin):
    """Central inference orchestrator driven by continuous attractor dynamics and MoE gating."""

    def __init__(
        self,
        ingestion_filter: MahalanobisFilter,
        rhythm_generator: RhythmTrajectoryGenerator,
        moe_gating: MultiplicativeMoEGating,
        expert_jury: Sequence[ExpertJuryPort],
        drift_sensors: Sequence[DriftSensorPort],
        outlier_reset_threshold: int = 3,
        exploration_boost_events: int = 5,
        state_store: MLStateStore | None = None,
        engine_id: str = "default",
        checkpoint_interval: int = 100,
        shadow_experts: Sequence[Any] = (),
    ) -> None:
        self._ingestion = ingestion_filter
        self._rhythm = rhythm_generator
        self._gating = moe_gating
        self._jury = list(expert_jury)
        self._shadow_experts = list(shadow_experts)
        self._sensors = list(drift_sensors)
        self._tracker = TrajectoryTracker()
        self._state_machine = StateMachine(outlier_reset_threshold=outlier_reset_threshold)
        self._state_store = state_store
        self._engine_id = engine_id
        self._checkpoint_interval = max(1, checkpoint_interval)
        self.outlier_reset_threshold = outlier_reset_threshold
        self.exploration_boost_events = exploration_boost_events

    @property
    def auto_reset_count(self) -> int:
        return self._state_machine.state.auto_resets

    @property
    def state_machine(self) -> StateMachine:
        return self._state_machine

    @property
    def _consecutive_outliers(self) -> int:
        return self._state_machine.state.consecutive_outliers

    @_consecutive_outliers.setter
    def _consecutive_outliers(self, value: int) -> None:
        self._state_machine.state.consecutive_outliers = value

    @property
    def _auto_resets(self) -> int:
        return self._state_machine.state.auto_resets

    @_auto_resets.setter
    def _auto_resets(self, value: int) -> None:
        self._state_machine.state.auto_resets = value

    def process_event(self, delta_state: np.ndarray, delta_time: float) -> ExecutionPlan:
        """Process an agnostic state transition ΔS -> ExecutionPlan."""
        self._state_machine.record_event_processed()
        if self._state_store and self._state_machine.state.total_events_processed % self._checkpoint_interval == 0:
            self.checkpoint()

        movement, is_outlier = self._ingestion.process_raw_step(delta_state, delta_time)
        if is_outlier:
            if self._state_machine.on_outlier_detected(is_consecutive=True):
                self._trigger_auto_regime_reset()
                return ExecutionPlan.HOLD(reason="Auto_Regime_Reset_Triggered", alert=True)
            return ExecutionPlan.HOLD(reason="Noise_Outlier_Blocked_By_Module_1", alert=True)

        if self._tracker.has_active_trajectory:
            status = self._tracker.evaluate_step(movement)
            if not status.is_valid:
                inv_step = self._tracker.active_trajectory.invalidation_step if self._tracker.active_trajectory else None
                self._tracker.set_active_trajectory(None)
                self._state_machine.on_deviation(status.reason)
                if self._state_machine.state.consecutive_outliers >= self.outlier_reset_threshold:
                    self._trigger_auto_regime_reset()
                    return ExecutionPlan.HOLD(reason="Auto_Regime_Reset_Triggered", alert=True)
                if inv_step is None or status.step_index < inv_step:
                    return ExecutionPlan.EMERGENCY_FLUSH(f"Reactive_Trajectory_Deviation_At_Step_{status.step_index}: {status.reason}")
                return ExecutionPlan.HOLD(reason=f"Trajectory_Deviation_At_Step_{status.step_index}")
            self._state_machine.on_valid_step()
            self._state_machine.on_step_advance()
        else:
            self._state_machine.on_valid_step()

        drift_scores = [s.get_drift_score() for s in self._sensors]
        current_drift = max(drift_scores) if drift_scores else 0.0
        top_k = self._rhythm.generate_candidate_trajectories(movement, current_drift)
        if not top_k:
            return ExecutionPlan.HOLD(reason="Insufficient_Trajectory_Density")
        
        lambda_t = self._rhythm._compute_lambda(self._rhythm._theta_manager.compute_entropy(self._rhythm._latest_state_key), current_drift)
        phi_ritmo = top_k[0].coherence_score if top_k else 0.0
        validation = self._gating.evaluate_and_veto(trajectories=top_k, jury=self._jury, lambda_t=lambda_t, phi_ritmo=phi_ritmo)

        phi_moe_base = validation.global_confidence
        phi_ritmo = validation.chosen_trajectory.coherence_score if validation.chosen_trajectory else 0.0
        lambda_clamped = max(0.0, min(1.0, lambda_t))
        phi_moe_final = max(0.0, min(1.0, phi_moe_base * (1.0 - lambda_clamped * (1.0 - phi_ritmo))))

        if validation.veto_triggered or validation.chosen_trajectory is None:
            self._tracker.set_active_trajectory(None)
            self._state_machine.on_validation_veto("All trajectories vetoed by critical expert")
            details = {
                "expert_name": validation.veto_details.expert_name,
                "score": validation.veto_details.score,
                "reason": validation.veto_details.reason,
            } if validation.veto_details else {"reason": "All trajectories vetoed"}
            return ExecutionPlan.HOLD(reason="Trajectory_Vetoed_By_Critical_Expert", details=details)

        thash = hashlib.sha256(delta_state.tobytes() + struct.pack('<d', float(delta_time))).hexdigest()[:16]
        trace = {"telemetry_hash": thash, "lambda_t": lambda_clamped, "phi_ritmo": phi_ritmo, "phi_moe": phi_moe_final, "phi_moe_base": phi_moe_base}
        envelope = validation.envelope or ActionEnvelope(magnitude=float(phi_moe_final), bounds={}, max_steps=len(validation.chosen_trajectory.movements), metadata={"decision_trace": trace}, decision_trace=trace)

        action = self._determine_action(phi_moe_final, validation.chosen_trajectory)
        if action == "HOLD":
            return ExecutionPlan.HOLD(reason="Phi_MoE_Below_Gamma_Exec", details={"reason": "Phi_MoE_Below_Gamma_Exec", "decision_trace": trace})
        if action == "EMERGENCY_FLUSH":
            return ExecutionPlan.EMERGENCY_FLUSH(f"Geometric_Threshold_Breach_Phi_MoE_{phi_moe_final:.3f}")

        self._tracker.set_active_trajectory(validation.chosen_trajectory, start_step=1)
        self._state_machine.on_trajectory_start(f"traj_{validation.chosen_trajectory.terminal_state.step_index}", validation.chosen_trajectory.invalidation_step)
        return ExecutionPlan.EXECUTE(trajectory=validation.chosen_trajectory, confidence=phi_moe_final, envelope=envelope, invalidation_step=validation.chosen_trajectory.invalidation_step)

    def _determine_action(self, phi_moe: float, trajectory: Any) -> str:
        """Continuous ensemble convergence action evaluator."""
        gamma_exec = getattr(self, "gamma_exec", 0.5)
        if phi_moe < gamma_exec:
            return "HOLD"
        geometric_threshold = getattr(self, "geometric_threshold", -0.1)
        if trajectory is not None and hasattr(trajectory, "movements") and len(trajectory.movements) > 1:
            directions = getattr(trajectory, "directions", None)
            if directions is not None and len(directions) > 1:
                dots = np.sum(directions[1:] * directions[:-1], axis=1)
                if float(np.min(dots)) < geometric_threshold:
                    return "EMERGENCY_FLUSH"
        return "EXECUTE"

    def update_feedback(self, actual_state: np.ndarray, predicted_state: np.ndarray) -> None:
        """Propagates multivariate continuous state distance feedback to sensors and jury."""
        act = np.asarray(actual_state, dtype=np.float64).flatten()
        pred = np.asarray(predicted_state, dtype=np.float64).flatten()
        act_metric = float(np.linalg.norm(act)) if act.size > 0 else 0.0
        pred_metric = float(np.linalg.norm(pred)) if pred.size > 0 else 0.0
        for sensor in self._sensors:
            sensor.update(act_metric, pred_metric)
        for expert in self._jury:
            expert.update_learning(act_metric, pred_metric)
