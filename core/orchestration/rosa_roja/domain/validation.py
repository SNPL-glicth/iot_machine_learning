"""ValidationResult and VetoDetails domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from .trajectory import Trajectory
from .execution import ActionEnvelope


@dataclass(frozen=True, slots=True)
class VetoDetails:
    """Details about which expert vetoed and why."""
    expert_name: str
    expert_type: str  # "critical" | "non_critical"
    score: float
    threshold: float
    reason: str


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of Module 3 MoE evaluation and gating."""
    chosen_trajectory: Optional[Trajectory]
    global_confidence: float           # Phi_MoE score in [0.0, 1.0]
    envelope: Optional[ActionEnvelope] # Action parameters for execution
    veto_triggered: bool
    veto_details: Optional[VetoDetails] = None
    all_scores: dict = field(default_factory=dict)  # expert_name -> score
    variance_penalty: float = 0.0
    lambda_t: float = 0.0              # Exploration factor from Module 2
    phi_ritmo: float = 0.0             # Phi_Ritmo from chosen trajectory