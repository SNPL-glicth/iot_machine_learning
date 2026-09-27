"""MRTEngine: Maxwell-Ramanujan-Tesla Inverse Kinematics and Phase Conjugate Engine.

Dual-Engine Architecture (Maxwell Conjugate Symmetry):
    - Rosa Roja (Observable 3D Pole): Forward Kinematics. Solves the trajectory
      propagation problem (E-field inertia, rhythm, and forward continuation).
    - MRT Engine (Conjugate 4D Pole): Inverse Kinematics. Acts as a Phase Conjugate Mirror
      (PCM), evaluating the restoring elastic rebound vector (-x) from 4D Ramanujan
      crystal dissipation and magnetic vorticity ‖∇ × B‖ under time-reversal symmetry.
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from iot_machine_learning.infrastructure.ml.engines.core.factory import register_engine
from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)
from .algorithms.mrt_pipeline import MRTPipeline, MRTExecutionResult


@register_engine("mrt")
class MRTEngine(PredictionEngine):
    """PredictionEngine computing the 4D conjugate phase recovery trajectory.

    Provides the negative pole (z2) for the Sovereign Hopf Fibration, resolving
    singularities through counter-cyclical elastic rebound.
    """

    def __init__(
        self,
        elasticity_rebound: float = 0.85,
        horizon_steps: int = 5,
        strain_epsilon: float = 1e-3,
        min_history_points: int = 5,
        gamma_dissipation: float = 0.15,
    ) -> None:
        self._min_points = max(4, min_history_points)
        self._pipeline = MRTPipeline(
            elasticity_rebound=elasticity_rebound,
            horizon_steps=horizon_steps,
            strain_epsilon=strain_epsilon,
            gamma_dissipation=gamma_dissipation,
        )

    @property
    def name(self) -> str:
        return "mrt"

    @property
    def pipeline(self) -> MRTPipeline:
        """Access the underlying conjugate processing pipeline."""
        return self._pipeline

    def can_handle(self, n_points: int) -> bool:
        """Determines if the observation window contains enough points for phase immersion."""
        return n_points >= self._min_points

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        """Evaluate phase-conjugate elastic rebound trajectory and 4D amplitude z2.

        Args:
            values: Chronological sequence of signal/market observations.
            timestamps: Optional timestamps (used to derive exact delta_time).

        Returns:
            PredictionResult containing the conjugate recovery vector and z2 confidence.
        """
        clean_v = [float(v) for v in values if v is not None and np.isfinite(v)]
        if not self.can_handle(len(clean_v)):
            fallback_val = clean_v[-1] if clean_v else 0.0
            return PredictionResult(
                predicted_value=fallback_val,
                confidence=0.0,
                trend="stable",
                metadata={"engine_name": "mrt", "status": "insufficient_history"},
            )

        dt = 1.0
        if timestamps is not None and len(timestamps) >= 2:
            dt_diff = float(timestamps[-1]) - float(timestamps[-2])
            if dt_diff > 1e-4:
                dt = dt_diff

        # Execute full conjugate pipeline
        exec_res: MRTExecutionResult = self._pipeline.process(clean_v, delta_time=dt)

        return PredictionResult(
            predicted_value=exec_res.rebound_target,
            confidence=exec_res.confidence,
            trend=exec_res.rebound_direction,  # type: ignore[arg-type]
            metadata={
                "engine_name": "mrt",
                "kinematics_mode": "inverse_conjugate_mirror",
                "z2_amplitude": exec_res.z2_amplitude,
                "theta2_phase": exec_res.theta2_phase,
                "vorticity_curl": exec_res.vorticity_curl,
                "frob_deformation_rate": exec_res.frob_deformation_rate,
                "rebound_trajectory": exec_res.rebound_trajectory,
                "rebound_magnitude": exec_res.rebound_magnitude,
                "ramanujan_trace": exec_res.ramanujan_trace,
                "dissipation_factor": exec_res.dissipation_factor,
                "status": exec_res.status,
                "evidence": exec_res.evidence,
            },
        )
