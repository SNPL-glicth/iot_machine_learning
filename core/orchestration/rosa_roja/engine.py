"""Rosa Roja Engine: Master System Orchestrator."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Sequence

import numpy as np

from .domain.movement import Movement
from .domain.execution import ExecutionPlan, ActionEnvelope
from .domain.trajectory_tracker import TrajectoryTracker, DeviationStatus
from .domain.ml_taxonomy import (
    PredictionConfidence,
    TrajectoryCoherence,
    MoEConfidence,
    ExecutionConfidence,
    DriftSeverity,
    TransitionEntropy,
    ExplorationFactor,
    ActionMagnitude,
)
from .domain.state_machine import StateMachine, PipelineState
from .domain.state_persistence import STATE_SCHEMA_VERSION
from .modules.module1_ingestion import MahalanobisFilter
from .modules.rhythm_generator import RhythmTrajectoryGenerator
from .modules.module3_moe_gating import MultiplicativeMoEGating
from .ports.expert_jury import ExpertJuryPort
from .ports.drift_sensor import DriftSensorPort
from .ports.state_store import MLStateStore

logger = logging.getLogger(__name__)


class RosaRojaEngine:
    """
    Master System Orchestrator driven by Stochastic Active Control, 
    Game Theory (Gungi-inspired position spaces), and Bayesian Active Inference.
    
    Controls raw event ingestion, trajectory generation, MoE hard-gating,
    and returns actionable ExecutionPlan objects.
    
    This is the CENTRAL CORE OF THE SYSTEM - it owns and orchestrates
    the Mixture of Experts (MoE) and Drift Detectors.
    """

    def __init__(
        self,
        ingestion_filter: MahalanobisFilter,
        rhythm_generator: RhythmTrajectoryGenerator,
        moe_gating: MultiplicativeMoEGating,
        expert_jury: Sequence[ExpertJuryPort],
        drift_sensors: Sequence[DriftSensorPort],
        outlier_reset_threshold: int = 3,
        exploration_boost_events: int = 5,
        state_store: Optional[MLStateStore] = None,
        engine_id: str = "default",
        checkpoint_interval: int = 100,
    ):
        self._ingestion = ingestion_filter
        self._rhythm = rhythm_generator
        self._gating = moe_gating
        self._jury = list(expert_jury)
        self._sensors = list(drift_sensors)
        self._tracker = TrajectoryTracker()

        # Canonical state machine
        self._state_machine = StateMachine(outlier_reset_threshold=outlier_reset_threshold)

        # Optional persistence: None disables it entirely (cold-start only).
        self._state_store = state_store
        self._engine_id = engine_id
        self._checkpoint_interval = max(1, checkpoint_interval)

        # Legacy fields for backward compatibility (deprecated)
        self.outlier_reset_threshold = outlier_reset_threshold
        self.exploration_boost_events = exploration_boost_events
        self._consecutive_outliers = 0
        self._auto_resets = 0

        # Sync state machine counters with legacy fields
        self._state_machine._state.consecutive_outliers = 0
        self._state_machine._state.auto_resets = 0

    @property
    def auto_reset_count(self) -> int:
        return self._state_machine.state.auto_resets

    @property
    def state_machine(self) -> StateMachine:
        """Access the canonical state machine."""
        return self._state_machine

    # Backward compatibility properties
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
        """
        Main entry point for processing a new state transition S_t -> S_{t+1}.
        
        Args:
            delta_state: State change vector ΔS (multidimensional)
            delta_time: Time delta Δt
            
        Returns:
            ExecutionPlan with action, trajectory, confidence, and risk parameters
        """
        # Record event
        self._state_machine.record_event_processed()

        # Write-behind checkpoint: bounded loss window (≤ checkpoint_interval
        # events). Runs at entry so every return path shares one hook.
        if (
            self._state_store is not None
            and self._state_machine.state.total_events_processed % self._checkpoint_interval == 0
        ):
            self.checkpoint()

        # 1. Module 1: Anti-Contamination Ingestion
        movement, is_outlier = self._ingestion.process_raw_step(delta_state, delta_time)
        
        if is_outlier:
            # Update state machine - returns True if regime reset was triggered
            reset_triggered = self._state_machine.on_outlier_detected(is_consecutive=True)
            
            if reset_triggered:
                self._trigger_auto_regime_reset()
                return ExecutionPlan.HOLD(
                    reason="Auto_Regime_Reset_Triggered",
                    alert=True
                )
            return ExecutionPlan.HOLD(
                reason="Noise_Outlier_Blocked_By_Module_1",
                alert=True
            )

        # 2. Reactive step monitoring against active trajectory.
        # Runs BEFORE the consecutive-outlier reset: a step that deviates from
        # the planned trajectory is not a valid step, so it must accumulate
        # with Module 1 outliers towards the regime-reset threshold.
        if self._tracker.has_active_trajectory:
            status = self._tracker.evaluate_step(movement)
            if not status.is_valid:
                # Capture invalidation info before clearing tracker
                active_traj = self._tracker.active_trajectory
                inv_step = active_traj.invalidation_step if active_traj else None
                self._tracker.set_active_trajectory(None)
                
                # Update state machine
                self._state_machine.on_deviation(status.reason)
                
                if self._state_machine.state.consecutive_outliers >= self.outlier_reset_threshold:
                    self._trigger_auto_regime_reset()
                    return ExecutionPlan.HOLD(
                        reason="Auto_Regime_Reset_Triggered",
                        alert=True
                    )
                
                # EMERGENCY_FLUSH if deviation before predicted invalidation (surprise)
                if inv_step is None or status.step_index < inv_step:
                    return ExecutionPlan.EMERGENCY_FLUSH(
                        f"Reactive_Trajectory_Deviation_At_Step_{status.step_index}: {status.reason}"
                    )
                return ExecutionPlan.HOLD(
                    reason=f"Trajectory_Deviation_At_Step_{status.step_index}"
                )
            
            # Valid tracked step - resets the consecutive outlier counter and
            # advances the tracker
            self._state_machine.on_valid_step()
            self._state_machine.on_step_advance()
        else:
            # Non-tracked non-outlier step resets the consecutive outlier counter
            self._state_machine.on_valid_step()

        # 3. Extract current aggregate drift score from sensors
        drift_scores = [s.get_drift_score() for s in self._sensors]
        current_drift = max(drift_scores) if drift_scores else 0.0

        # 4. Module 2: Trajectory & Rhythm Density Generation
        top_k_trajectories = self._rhythm.generate_candidate_trajectories(
            latest_movement=movement,
            drift_score=current_drift
        )

        if not top_k_trajectories:
            return ExecutionPlan.HOLD(reason="Insufficient_Trajectory_Density")

        # 5. Module 3: MoE Coherence, Critical Veto & Variance Penalty
        validation = self._gating.evaluate_and_veto(
            trajectories=top_k_trajectories,
            jury=self._jury
        )
        
        if validation.veto_triggered or validation.chosen_trajectory is None:
            self._tracker.set_active_trajectory(None)
            self._state_machine.on_validation_veto("All trajectories vetoed by critical expert")
            return ExecutionPlan.HOLD(
                reason="Trajectory_Vetoed_By_Critical_Expert",
                details=(
                    {
                        "expert_name": validation.veto_details.expert_name,
                        "expert_type": validation.veto_details.expert_type,
                        "score": validation.veto_details.score,
                        "threshold": validation.veto_details.threshold,
                        "reason": validation.veto_details.reason,
                    }
                    if validation.veto_details else {}
                )
            )
        
        # 6. Build Final Orchestrated Execution Plan
        envelope = validation.envelope
        
        # Set new active trajectory for reactive monitoring (start at step 1: next movement)
        self._tracker.set_active_trajectory(
            validation.chosen_trajectory, 
            start_step=1
        )
        
        # Update state machine
        trajectory_id = f"traj_{validation.chosen_trajectory.terminal_state.step_index}"
        self._state_machine.on_trajectory_start(
            trajectory_id=trajectory_id,
            invalidation_step=validation.chosen_trajectory.invalidation_step
        )
        
        return ExecutionPlan.EXECUTE(
            trajectory=validation.chosen_trajectory,
            confidence=validation.global_confidence,
            envelope=envelope,
            invalidation_step=validation.chosen_trajectory.invalidation_step
        )

    def _trigger_auto_regime_reset(self) -> None:
        """Automatic regime recovery: full reset of Module 1 covariance, boost exploration."""
        self._ingestion.reset()
        self._rhythm.boost_exploration(self.exploration_boost_events)
        self._tracker.set_active_trajectory(None)
        self._consecutive_outliers = 0
        # State machine's trigger_regime_reset was already called by on_outlier_detected
        # and increments the auto_resets counter there

    def update_feedback(self, actual_state: np.ndarray, predicted_state: np.ndarray) -> None:
        """
        Propagates actual results to drift sensors and jury experts for learning.
        Called after actual outcome is known.
        """
        # Use first dimension for scalar feedback (price/value)
        actual_scalar = float(actual_state[0]) if len(actual_state) > 0 else 0.0
        predicted_scalar = float(predicted_state[0]) if len(predicted_state) > 0 else 0.0
        
        for sensor in self._sensors:
            sensor.update(actual_scalar, predicted_scalar)
        
        for expert in self._jury:
            expert.update_learning(actual_scalar, predicted_scalar)

    def register_expert(self, expert: ExpertJuryPort) -> None:
        """Dynamically register a new expert to the jury."""
        if expert not in self._jury:
            self._jury.append(expert)

    def register_drift_sensor(self, sensor: DriftSensorPort) -> None:
        """Dynamically register a new drift sensor."""
        if sensor not in self._sensors:
            self._sensors.append(sensor)

    def get_jury_status(self) -> dict:
        """Get status of all jury experts."""
        return {
            expert.name: {
                "is_critical": expert.is_critical,
                "threshold": expert.threshold,
                "weight": expert.weight,
            }
            for expert in self._jury
        }

    def get_drift_status(self) -> dict:
        """Get status of all drift sensors."""
        return {
            sensor.name: {
                "drift_score": sensor.get_drift_score(),
            }
            for sensor in self._sensors
        }

    def get_state_summary(self) -> dict:
        """Get canonical state machine summary."""
        return self._state_machine.get_state_summary()

    # ------------------------------------------------------------------
    # Persistence (warm start / recoverability)
    # ------------------------------------------------------------------

    def export_state(self) -> Dict[str, Any]:
        """Atomic snapshot of all learning state.

        The active trajectory is intentionally excluded: restoring with a
        flushed tracker yields a safe HOLD on the first event instead of
        validating against stale predictions. Jury experts and drift sensors
        that implement the StatePersistable contract are included keyed by
        name; members without the contract export a null state.
        """
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "engine_id": self._engine_id,
            "event_watermark": self._state_machine.state.total_events_processed,
            "saved_at": time.time(),
            "components": {
                "ingestion": self._ingestion.export_state(),
                "rhythm_generator": self._rhythm.export_state(),
                "state_machine": self._state_machine.export_state(),
                "jury": [
                    {"name": e.name, "state": e.export_state()}
                    for e in self._jury
                    if hasattr(e, "export_state")
                ],
                "sensors": [
                    {"name": s.name, "state": s.export_state()}
                    for s in self._sensors
                    if hasattr(s, "export_state")
                ],
            },
        }

    def import_state(self, payload: Dict[str, Any]) -> None:
        """Restore learning state from an export_state snapshot.

        Raises ValueError on unknown schemas or malformed payloads; callers
        should treat that as a cold-start signal.
        """
        if not isinstance(payload, dict) or payload.get("schema_version") != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported engine snapshot schema: "
                f"{payload.get('schema_version') if isinstance(payload, dict) else type(payload)!r}"
            )
        components = payload.get("components")
        if not isinstance(components, dict):
            raise ValueError("Engine snapshot missing 'components'")

        for required in ("ingestion", "rhythm_generator", "state_machine"):
            if required not in components:
                raise ValueError(f"Engine snapshot missing component: {required}")

        self._ingestion.import_state(components["ingestion"])
        self._rhythm.import_state(components["rhythm_generator"])
        self._state_machine.import_state(components["state_machine"])
        self._tracker.set_active_trajectory(None)

        # Restore persistable jury experts and drift sensors by name.
        # A saved member with no live counterpart is a composition mismatch
        # (loud failure); a live member absent from the snapshot simply
        # starts cold (dynamic registration stays possible).
        for key, members in (
            ("jury", self._jury),
            ("sensors", self._sensors),
        ):
            saved = components.get(key, [])
            if not isinstance(saved, list):
                raise ValueError(f"Engine snapshot '{key}' must be a list")
            live_by_name = {
                m.name: m for m in members if hasattr(m, "import_state")
            }
            for entry in saved:
                if not isinstance(entry, dict) or "name" not in entry:
                    raise ValueError(f"Malformed entry in engine snapshot '{key}'")
                name = entry["name"]
                member = live_by_name.get(name)
                if member is None:
                    raise ValueError(
                        f"Snapshot '{key}' member '{name}' has no live counterpart"
                    )
                if entry.get("state") is not None:
                    member.import_state(entry["state"])

    def checkpoint(self) -> bool:
        """Persist a snapshot to the configured store. False if disabled/failed."""
        if self._state_store is None:
            return False
        ok = self._state_store.save(self._engine_id, self.export_state())
        if not ok:
            logger.warning(
                "Checkpoint failed for engine %s; continuing in-memory", self._engine_id
            )
        return ok

    def restore(self, state_store: Optional[MLStateStore] = None) -> bool:
        """Warm start from the store. Returns True when state was restored.

        Any storage or schema failure degrades to cold start (False) without
        raising: availability of the pipeline never depends on the store.
        """
        store = state_store if state_store is not None else self._state_store
        if store is None:
            return False
        payload = store.load(self._engine_id)
        if payload is None:
            logger.info("No ML snapshot for %s; starting cold", self._engine_id)
            return False
        try:
            self.import_state(payload)
        except ValueError as exc:
            logger.error(
                "Corrupt ML snapshot for %s (%s); starting cold", self._engine_id, exc
            )
            return False
        logger.info(
            "Warm start for %s from watermark=%s",
            self._engine_id,
            payload.get("event_watermark"),
        )
        return True

    def reset(self) -> None:
        """Reset all internal state (e.g., after confirmed regime change)."""
        self._ingestion.reset()
        self._rhythm.reset()
        self._tracker.reset()
        self._consecutive_outliers = 0
        self._auto_resets = 0
        for sensor in self._sensors:
            sensor.reset()
        self._state_machine.reset()