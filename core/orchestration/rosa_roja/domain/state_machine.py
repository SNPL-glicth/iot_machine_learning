"""Canonical State Machine for Rosa Roja Pipeline.

Unifies all fragmented state representations into a single source of truth.
"""

from __future__ import annotations

from enum import Enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time

from .state_persistence import STATE_SCHEMA_VERSION


class PipelineState(Enum):
    """Main pipeline state - single source of truth for engine lifecycle."""
    COLD_START = "cold_start"           # No history, no trajectory
    WARMING = "warming"                 # Building covariance/history
    ACTIVE_TRACKING = "active_tracking" # Valid trajectory being monitored
    GENERATING = "generating"           # Module 2 producing candidates
    VALIDATING = "validating"           # Module 3 evaluating trajectories
    EXECUTING = "executing"             # Action dispatched
    REGIME_RESET = "regime_reset"       # Auto-reset triggered
    EMERGENCY = "emergency"             # Emergency flush active


class IngestionState(Enum):
    """Module 1: Mahalanobis ingestion filter state."""
    UNINITIALIZED = "uninitialized"
    WARMING = "warming"                 # < min_samples_for_cov
    ACTIVE = "active"                   # Covariance stable
    RESET_PENDING = "reset_pending"     # Auto-reset triggered


class TrackingState(Enum):
    """Module 2/Tracker: Trajectory monitoring state."""
    IDLE = "idle"
    TRACKING = "tracking"
    DEVIATED = "deviated"               # Step invalidated
    EXPIRED = "expired"                 # Trajectory completed


class GenerationState(Enum):
    """Module 2: Rhythm generator state."""
    INSUFFICIENT_DATA = "insufficient"
    LEARNING = "learning"               # Building transition graph
    GENERATING = "generating"
    BOOSTED = "boosted"                 # Exploration boost active


class ValidationState(Enum):
    """Module 3: MoE gating validation state."""
    IDLE = "idle"
    VETOED = "vetoed"                   # Critical expert vetoed all
    VALIDATED = "validated"             # Trajectory passed all gates


@dataclass
class StateTransition:
    """Record of a state transition for audit trail."""
    from_state: Enum
    to_state: Enum
    timestamp: float
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemState:
    """Complete system state snapshot."""
    # Main pipeline state
    pipeline_state: PipelineState = PipelineState.COLD_START
    
    # Sub-component states (orthogonal)
    ingestion_state: IngestionState = IngestionState.UNINITIALIZED
    tracking_state: TrackingState = TrackingState.IDLE
    generation_state: GenerationState = GenerationState.INSUFFICIENT_DATA
    validation_state: ValidationState = ValidationState.IDLE
    
    # Counters for observability
    consecutive_outliers: int = 0
    auto_resets: int = 0
    total_events_processed: int = 0
    
    # Current trajectory info (if tracking)
    active_trajectory_id: Optional[str] = None
    current_trajectory_step: int = 0
    trajectory_invalidation_step: Optional[int] = None
    
    # Timestamp of last state change
    last_transition_time: float = field(default_factory=time.time)
    
    # Audit trail (last 100 transitions)
    transition_history: list[StateTransition] = field(default_factory=list)


class StateMachine:
    """Canonical state machine for Rosa Roja engine.
    
    Single source of truth for all state transitions.
    """
    
    def __init__(self, outlier_reset_threshold: int = 3):
        self._state = SystemState()
        self._outlier_reset_threshold = outlier_reset_threshold
        self._max_history = 100
    
    @property
    def state(self) -> SystemState:
        return self._state
    
    @property
    def pipeline_state(self) -> PipelineState:
        return self._state.pipeline_state
    
    def transition(
        self,
        new_pipeline_state: Optional[PipelineState] = None,
        new_ingestion_state: Optional[IngestionState] = None,
        new_tracking_state: Optional[TrackingState] = None,
        new_generation_state: Optional[GenerationState] = None,
        new_validation_state: Optional[ValidationState] = None,
        reason: str = "",
        **metadata
    ) -> None:
        """Perform a state transition with audit logging."""
        old_pipeline = self._state.pipeline_state
        old_ingestion = self._state.ingestion_state
        old_tracking = self._state.tracking_state
        old_generation = self._state.generation_state
        old_validation = self._state.validation_state
        
        # Update states if provided
        if new_pipeline_state is not None:
            self._state.pipeline_state = new_pipeline_state
        if new_ingestion_state is not None:
            self._state.ingestion_state = new_ingestion_state
        if new_tracking_state is not None:
            self._state.tracking_state = new_tracking_state
        if new_generation_state is not None:
            self._state.generation_state = new_generation_state
        if new_validation_state is not None:
            self._state.validation_state = new_validation_state
        
        # Log transition if any state changed
        if (new_pipeline_state and new_pipeline_state != old_pipeline) or \
           (new_ingestion_state and new_ingestion_state != old_ingestion) or \
           (new_tracking_state and new_tracking_state != old_tracking) or \
           (new_generation_state and new_generation_state != old_generation) or \
           (new_validation_state and new_validation_state != old_validation):
            
            timestamp = time.time()
            self._state.last_transition_time = timestamp
            
            # Record primary pipeline transition
            if new_pipeline_state and new_pipeline_state != old_pipeline:
                self._record_transition(old_pipeline, new_pipeline_state, timestamp, reason, metadata)
            
            # Record sub-component transitions
            if new_ingestion_state and new_ingestion_state != old_ingestion:
                self._record_transition(old_ingestion, new_ingestion_state, timestamp, reason, metadata)
            if new_tracking_state and new_tracking_state != old_tracking:
                self._record_transition(old_tracking, new_tracking_state, timestamp, reason, metadata)
            if new_generation_state and new_generation_state != old_generation:
                self._record_transition(old_generation, new_generation_state, timestamp, reason, metadata)
            if new_validation_state and new_validation_state != old_validation:
                self._record_transition(old_validation, new_validation_state, timestamp, reason, metadata)
    
    def _record_transition(
        self,
        from_state: Enum,
        to_state: Enum,
        timestamp: float,
        reason: str,
        metadata: Dict[str, Any]
    ) -> None:
        transition = StateTransition(
            from_state=from_state,
            to_state=to_state,
            timestamp=timestamp,
            reason=reason,
            metadata=metadata
        )
        self._state.transition_history.append(transition)
        if len(self._state.transition_history) > self._max_history:
            self._state.transition_history.pop(0)
    
    def on_outlier_detected(self, is_consecutive: bool = True) -> bool:
        """Handle Module 1 outlier detection.
        
        Returns:
            True if a regime reset was triggered, False otherwise.
        """
        if is_consecutive:
            self._state.consecutive_outliers += 1
        else:
            self._state.consecutive_outliers = 1
        
        self._state.ingestion_state = IngestionState.ACTIVE  # Still active, just flagged
        
        if self._state.consecutive_outliers >= self._outlier_reset_threshold:
            self.trigger_regime_reset()
            return True
        
        return False
    
    def on_valid_step(self) -> None:
        """Handle valid (non-outlier) step."""
        self._state.consecutive_outliers = 0
    
    def on_trajectory_start(self, trajectory_id: str, invalidation_step: int) -> None:
        """Handle new active trajectory."""
        self._state.active_trajectory_id = trajectory_id
        self._state.current_trajectory_step = 0
        self._state.trajectory_invalidation_step = invalidation_step
        self._state.tracking_state = TrackingState.TRACKING
        self._state.pipeline_state = PipelineState.ACTIVE_TRACKING
    
    def on_step_advance(self) -> None:
        """Advance trajectory step counter."""
        self._state.current_trajectory_step += 1
    
    def on_deviation(self, reason: str) -> None:
        """Handle trajectory deviation."""
        self._state.tracking_state = TrackingState.DEVIATED
        self._state.pipeline_state = PipelineState.EMERGENCY
        # Increment consecutive outliers counter for deviation
        self._state.consecutive_outliers += 1
        self.transition(
            new_pipeline_state=PipelineState.EMERGENCY,
            new_tracking_state=TrackingState.DEVIATED,
            reason=f"trajectory_deviation: {reason}"
        )
    
    def on_trajectory_complete(self) -> None:
        """Handle trajectory completion."""
        self._state.tracking_state = TrackingState.EXPIRED
        self._state.active_trajectory_id = None
        self._state.current_trajectory_step = 0
        self._state.trajectory_invalidation_step = None
        self._state.pipeline_state = PipelineState.GENERATING
    
    def on_generation_start(self) -> None:
        """Start trajectory generation."""
        self._state.generation_state = GenerationState.GENERATING
        self._state.pipeline_state = PipelineState.GENERATING
    
    def on_generation_boosted(self) -> None:
        """Exploration boost activated."""
        self._state.generation_state = GenerationState.BOOSTED
    
    def on_validation_veto(self, reason: str) -> None:
        """MoE vetoed all trajectories."""
        self._state.validation_state = ValidationState.VETOED
        self._state.pipeline_state = PipelineState.REGIME_RESET
    
    def on_validation_pass(self) -> None:
        """MoE validated a trajectory."""
        self._state.validation_state = ValidationState.VALIDATED
        self._state.pipeline_state = PipelineState.EXECUTING
    
    def trigger_regime_reset(self) -> None:
        """Trigger automatic regime reset."""
        self._state.auto_resets += 1
        self._state.consecutive_outliers = 0
        self._state.ingestion_state = IngestionState.RESET_PENDING
        self._state.pipeline_state = PipelineState.REGIME_RESET
        self._state.generation_state = GenerationState.BOOSTED
        self.transition(
            new_pipeline_state=PipelineState.REGIME_RESET,
            new_ingestion_state=IngestionState.RESET_PENDING,
            new_generation_state=GenerationState.BOOSTED,
            reason="auto_regime_reset"
        )
    
    def trigger_emergency_flush(self, reason: str) -> None:
        """Trigger emergency flush."""
        self.transition(
            new_pipeline_state=PipelineState.EMERGENCY,
            new_validation_state=ValidationState.VETOED,
            reason=f"emergency_flush: {reason}"
        )
    
    def on_execution_dispatched(self) -> None:
        """Action dispatched to execution layer."""
        self._state.pipeline_state = PipelineState.EXECUTING
    
    def record_event_processed(self) -> None:
        """Increment event counter."""
        self._state.total_events_processed += 1
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get serializable state summary."""
        return {
            "pipeline_state": self._state.pipeline_state.value,
            "ingestion_state": self._state.ingestion_state.value,
            "tracking_state": self._state.tracking_state.value,
            "generation_state": self._state.generation_state.value,
            "validation_state": self._state.validation_state.value,
            "consecutive_outliers": self._state.consecutive_outliers,
            "auto_resets": self._state.auto_resets,
            "total_events": self._state.total_events_processed,
            "active_trajectory": self._state.active_trajectory_id,
            "trajectory_step": self._state.current_trajectory_step,
            "invalidation_step": self._state.trajectory_invalidation_step,
            "last_transition": self._state.last_transition_time,
        }
    
    def reset(self) -> None:
        """Full state reset."""
        self._state = SystemState()

    # ------------------------------------------------------------------
    # Persistence (StatePersistable contract)
    # ------------------------------------------------------------------

    _ENUM_REGISTRY: Dict[str, type] = {
        cls.__name__: cls
        for cls in (
            PipelineState,
            IngestionState,
            TrackingState,
            GenerationState,
            ValidationState,
        )
    }

    def export_state(self) -> Dict[str, Any]:
        """Serialize counters, sub-states and the audit trail tail."""
        s = self._state
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "pipeline_enum": "PipelineState",
            "pipeline_state": s.pipeline_state.value,
            "ingestion_enum": "IngestionState",
            "ingestion_state": s.ingestion_state.value,
            "tracking_enum": "TrackingState",
            "tracking_state": s.tracking_state.value,
            "generation_enum": "GenerationState",
            "generation_state": s.generation_state.value,
            "validation_enum": "ValidationState",
            "validation_state": s.validation_state.value,
            "consecutive_outliers": s.consecutive_outliers,
            "auto_resets": s.auto_resets,
            "total_events_processed": s.total_events_processed,
            "active_trajectory_id": s.active_trajectory_id,
            "current_trajectory_step": s.current_trajectory_step,
            "trajectory_invalidation_step": s.trajectory_invalidation_step,
            "last_transition_time": s.last_transition_time,
            "transition_history": [
                {
                    "from_enum": t.from_state.__class__.__name__,
                    "from_value": t.from_state.value,
                    "to_enum": t.to_state.__class__.__name__,
                    "to_value": t.to_state.value,
                    "timestamp": t.timestamp,
                    "reason": t.reason,
                    "metadata": dict(t.metadata),
                }
                for t in s.transition_history
            ],
        }

    def import_state(self, payload: Dict[str, Any]) -> None:
        """Restore counters, sub-states and audit trail from a snapshot.

        Values are assigned directly: restoring must not re-trigger
        transition logic or emit synthetic audit entries.
        """
        if not isinstance(payload, dict) or "schema_version" not in payload:
            raise ValueError("StateMachine payload missing schema_version")
        if payload["schema_version"] != STATE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported StateMachine schema: {payload['schema_version']}"
            )

        def _parse(name_key: str, value_key: str) -> Enum:
            enum_cls = self._ENUM_REGISTRY.get(payload.get(name_key, ""))
            if enum_cls is None:
                raise ValueError(f"Unknown enum class {payload.get(name_key)!r}")
            try:
                return enum_cls(payload[value_key])
            except (KeyError, ValueError, TypeError):
                raise ValueError(f"Invalid {name_key}={payload.get(value_key)!r}")

        history_raw = payload.get("transition_history", [])
        if not isinstance(history_raw, list):
            raise ValueError("StateMachine payload 'transition_history' must be a list")

        restored_history = []
        for entry in history_raw:
            try:
                from_cls = self._ENUM_REGISTRY[entry["from_enum"]]
                to_cls = self._ENUM_REGISTRY[entry["to_enum"]]
                restored_history.append(
                    StateTransition(
                        from_state=from_cls(entry["from_value"]),
                        to_state=to_cls(entry["to_value"]),
                        timestamp=float(entry["timestamp"]),
                        reason=str(entry.get("reason", "")),
                        metadata=dict(entry.get("metadata") or {}),
                    )
                )
            except (KeyError, ValueError, TypeError):
                raise ValueError("Malformed transition entry in audit trail")

        self.reset()
        s = self._state
        s.pipeline_state = _parse("pipeline_enum", "pipeline_state")
        s.ingestion_state = _parse("ingestion_enum", "ingestion_state")
        s.tracking_state = _parse("tracking_enum", "tracking_state")
        s.generation_state = _parse("generation_enum", "generation_state")
        s.validation_state = _parse("validation_enum", "validation_state")

        s.consecutive_outliers = int(payload.get("consecutive_outliers", 0))
        s.auto_resets = int(payload.get("auto_resets", 0))
        s.total_events_processed = int(payload.get("total_events_processed", 0))
        active_id = payload.get("active_trajectory_id")
        s.active_trajectory_id = str(active_id) if active_id is not None else None
        step = payload.get("current_trajectory_step")
        s.current_trajectory_step = int(step) if step is not None else 0
        inv = payload.get("trajectory_invalidation_step")
        s.trajectory_invalidation_step = int(inv) if inv is not None else None
        ltt = payload.get("last_transition_time")
        s.last_transition_time = float(ltt) if ltt is not None else time.time()
        s.transition_history.extend(restored_history[-self._max_history:])