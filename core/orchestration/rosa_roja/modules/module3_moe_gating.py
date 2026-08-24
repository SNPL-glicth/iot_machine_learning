"""Module 3: MoE Coherence & Hard-Gating Veto."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np

from ..domain.trajectory import Trajectory
from ..domain.validation import ValidationResult, VetoDetails
from ..ports.expert_jury import ExpertJuryPort
from ..domain.execution import ActionEnvelope


@dataclass
class MultiplicativeMoEGating:
    """
    MoE Coherence & Hard-Gating Veto.
    
    Evaluates trajectories across MoE experts. Applies a strict multiplicative 
    veto (I=0) if any critical expert fails its threshold τ_k, and penalizes 
    trajectory confidence proportional to inter-expert variance Var(Ψ_e).
    
    Equation: Φ_MoE(T) = [∏_{k∈Críticos} I(Ψ_k(T) ≥ τ_k)] · (Σ w_e·Ψ_e(T)) / (1 + γ·Var({Ψ_e(T)}))
    """
    
    variance_penalty: float = 0.5        # γ in denominator
    
    # Confidence bands for action envelope calculation
    # (low, high, magnitude, stop_pct, target_pct, max_steps)
    CONFIDENCE_BANDS = [
        (0.9, 1.0, 1.0, 0.015, 0.06, 20),   # Very high
        (0.7, 0.9, 0.7, 0.020, 0.05, 15),   # High
        (0.5, 0.7, 0.4, 0.025, 0.04, 12),   # Medium
        (0.3, 0.5, 0.2, 0.030, 0.03, 10),   # Low
        (0.0, 0.3, 0.0, 0.000, 0.00, 0),    # No action
    ]
    
    def evaluate_and_veto(
        self, 
        trajectories: list[Trajectory], 
        jury: Sequence[ExpertJuryPort]
    ) -> ValidationResult:
        """
        Evaluate all trajectories through MoE jury with hard-gating veto.
        
        Returns:
            ValidationResult with chosen trajectory or veto details
        """
        best_trajectory = None
        best_score = -1.0
        best_all_scores = {}
        veto_details = None
        
        for traj in trajectories:
            # Phase 1: Hard gating - critical experts must pass threshold
            veto = self._check_critical_veto(traj, jury)
            if veto:
                # Track the first veto for reporting
                if veto_details is None:
                    veto_details = veto
                continue
            
            # Phase 2: Soft scoring with variance penalty
            scores = [expert.evaluate_trajectory(traj) for expert in jury]
            weights = [expert.weight for expert in jury]
            
            weighted_mean = np.average(scores, weights=weights)
            variance = np.var(scores)
            phi_moe = weighted_mean / (1 + self.variance_penalty * variance)
            
            if phi_moe > best_score:
                best_score = phi_moe
                best_trajectory = traj
                best_all_scores = {expert.name: score for expert, score in zip(jury, scores)}
        
        if best_trajectory is None:
            return ValidationResult(
                chosen_trajectory=None,
                global_confidence=0.0,
                envelope=None,
                veto_triggered=True,
                veto_details=veto_details or VetoDetails(
                    expert_name="unknown",
                    expert_type="critical",
                    score=0.0,
                    threshold=0.0,
                    reason="All trajectories vetoed"
                ),
            )
        
        # Calculate action envelope from confidence
        envelope = self._confidence_to_envelope(best_score)
        
        return ValidationResult(
            chosen_trajectory=best_trajectory,
            global_confidence=best_score,
            envelope=envelope,
            veto_triggered=False,
            veto_details=None,
            all_scores=best_all_scores,
            variance_penalty=variance,
        )
    
    def _check_critical_veto(
        self, 
        trajectory: Trajectory, 
        jury: Sequence[ExpertJuryPort]
    ) -> Optional[VetoDetails]:
        """
        Check if any critical expert vetoes this trajectory.
        Returns VetoDetails if vetoed, None otherwise.
        """
        for expert in jury:
            if expert.is_critical:
                score = expert.evaluate_trajectory(trajectory)
                if score < expert.threshold:
                    return VetoDetails(
                        expert_name=expert.name,
                        expert_type="critical",
                        score=score,
                        threshold=expert.threshold,
                        reason=f"Critical expert {expert.name} scored {score:.3f} below threshold {expert.threshold}"
                    )
        return None
    
    def _confidence_to_envelope(self, confidence: float) -> ActionEnvelope:
        """Map global confidence to action envelope."""
        for low, high, magnitude, stop_pct, target_pct, max_steps in self.CONFIDENCE_BANDS:
            if low <= confidence < high:
                return ActionEnvelope(
                    magnitude=magnitude,
                    bounds={"stop_pct": stop_pct, "target_pct": target_pct},
                    max_steps=max_steps,
                    metadata={"confidence_band": f"{low}-{high}"}
                )
        return ActionEnvelope(0.0, {}, 0, {})