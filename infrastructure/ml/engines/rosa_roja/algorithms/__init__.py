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
from domain.entities.rosa_roja.execution import ExecutionPlan, ActionEnvelope
from domain.entities.rosa_roja.movement import Movement, RhythmSignature
from domain.entities.rosa_roja.trajectory import Trajectory, TerminalState
from domain.entities.rosa_roja.validation import ValidationResult, VetoDetails
from .modules.module1_ingestion import MahalanobisFilter
from .modules.rhythm_generator import RhythmTrajectoryGenerator
from .modules.module3_moe_gating import MultiplicativeMoEGating
from domain.ports.rosa_roja.expert_jury import ExpertJuryPort
from domain.ports.rosa_roja.drift_sensor import DriftSensorPort
from domain.ports.rosa_roja.execution_port import ExecutionPort

import sys
import domain.entities.rosa_roja as _rr_domain
import domain.ports.rosa_roja as _rr_ports

# Register dynamic aliases for backward compatibility
for prefix in ("infrastructure.ml.engines.rosa_roja.algorithms", "iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms"):
    sys.modules.setdefault(f"{prefix}.domain", _rr_domain)
    sys.modules.setdefault(f"{prefix}.ports", _rr_ports)
    for mod_name in ("movement", "trajectory", "execution", "theta_belief", "validation", "trajectory_tracker", "state_machine", "state_persistence", "engine_persistence", "ml_taxonomy"):
        submod = sys.modules.get(f"domain.entities.rosa_roja.{mod_name}") or getattr(_rr_domain, mod_name, None)
        if submod:
            sys.modules.setdefault(f"{prefix}.domain.{mod_name}", submod)
    for port_name in ("execution_port", "drift_sensor", "expert_jury", "state_store"):
        submod = sys.modules.get(f"domain.ports.rosa_roja.{port_name}") or getattr(_rr_ports, port_name, None)
        if submod:
            sys.modules.setdefault(f"{prefix}.ports.{port_name}", submod)

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