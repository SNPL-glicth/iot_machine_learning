"""Domain models for Rosa Roja Engine."""

from .movement import Movement, RhythmSignature
from .trajectory import Trajectory, TerminalState
from .validation import ValidationResult, VetoDetails
from .execution import ExecutionPlan, ActionEnvelope
from .motif import TopologicalMotifKey
from .theta_belief import StateKey, StateKeyUnion, ThetaBelief
from .trajectory_tracker import DeviationStatus, TrajectoryTracker
from .ml_taxonomy import (
    PredictionConfidence,
    TrajectoryCoherence,
    MoEConfidence,
    ExecutionConfidence,
    DriftSeverity,
    TransitionEntropy,
    ExplorationFactor,
    ActionMagnitude,
    PositionCovariance,
    PredictionVariance,
    AnomalySeverity,
)
from .state_machine import (
    PipelineState,
    IngestionState,
    TrackingState,
    GenerationState,
    ValidationState,
    StateMachine,
    SystemState,
    StateTransition,
)

__all__ = [
    "Movement",
    "RhythmSignature",
    "Trajectory",
    "TerminalState",
    "ValidationResult",
    "VetoDetails",
    "ExecutionPlan",
    "ActionEnvelope",
    "TopologicalMotifKey",
    "StateKey",
    "StateKeyUnion",
    "ThetaBelief",
    "DeviationStatus",
    "TrajectoryTracker",
    # Unified Taxonomy
    "PredictionConfidence",
    "TrajectoryCoherence",
    "MoEConfidence",
    "ExecutionConfidence",
    "DriftSeverity",
    "TransitionEntropy",
    "ExplorationFactor",
    "ActionMagnitude",
    "PositionCovariance",
    "PredictionVariance",
    "AnomalySeverity",
    # State Machine
    "PipelineState",
    "IngestionState",
    "TrackingState",
    "GenerationState",
    "ValidationState",
    "StateMachine",
    "SystemState",
    "StateTransition",
]