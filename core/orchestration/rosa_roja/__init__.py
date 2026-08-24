"""Rosa Roja Engine: Master System Orchestrator.

Main exports:
- RosaRojaEngine: The central orchestrator
- ExecutionPlan: Final output for execution layer
- MahalanobisFilter: Module 1 (ingestion)
- RhythmTrajectoryGenerator: Module 2 (trajectory generation)
- MultiplicativeMoEGating: Module 3 (MoE gating)
- ExpertJuryPort: Protocol for MoE experts
- DriftSensorPort: Protocol for drift detectors
- ExecutionPort: Native protocol for execution handlers
"""

from .engine import RosaRojaEngine
from .domain.execution import ExecutionPlan, ActionEnvelope
from .domain.movement import Movement, RhythmSignature
from .domain.trajectory import Trajectory, TerminalState
from .domain.validation import ValidationResult, VetoDetails
from .modules.module1_ingestion import MahalanobisFilter
from .modules.rhythm_generator import RhythmTrajectoryGenerator
from .modules.module3_moe_gating import MultiplicativeMoEGating
from .ports.expert_jury import ExpertJuryPort
from .ports.drift_sensor import DriftSensorPort
from .ports.execution_port import ExecutionPort

__all__ = [
    "RosaRojaEngine",
    "ExecutionPlan",
    "ActionEnvelope",
    "Movement",
    "RhythmSignature",
    "Trajectory",
    "TerminalState",
    "ValidationResult",
    "VetoDetails",
    "MahalanobisFilter",
    "RhythmTrajectoryGenerator",
    "MultiplicativeMoEGating",
    "ExpertJuryPort",
    "DriftSensorPort",
    "ExecutionPort",
]

__version__ = "1.0.0"