"""Takens Topological Inference engine adapter for Rosa Roja ExpertJury."""

from __future__ import annotations

from typing import Any
import numpy as np

from domain.entities.rosa_roja.trajectory import Trajectory
from domain.entities.takens.takens_parameters import TakensParameters
from domain.entities.takens.topological_audit import TopologicalAuditRecord
from infrastructure.ml.engines.ramanujan_takens.engine import RamanujanTakensEngine
from infrastructure.ml.interfaces import PredictionEngine
from .base_adapter import BaseExpertAdapter


class TakensExpertAdapter(BaseExpertAdapter):
    """Adapter for Ramanujan Takens Topological Inference engine into Rosa Roja MoE Jury.

    Acts as a specialized topological expert capable of detecting hidden attractor
    instabilities, false nearest neighbors folding, and issuing hard critical vetoes.
    """

    def __init__(
        self,
        engine: PredictionEngine | None = None,
        is_critical: bool = True,
        threshold: float = 0.50,
        weight: float = 1.0,
        params: TakensParameters | None = None,
    ) -> None:
        """Initialize adapter wrapping RamanujanTakensEngine as an ExpertJuryPort."""
        actual_engine = engine or RamanujanTakensEngine(params=params)
        super().__init__(
            engine=actual_engine,
            name="ramanujan_takens",
            is_critical=is_critical,
            threshold=threshold,
            weight=weight,
        )
        self._latest_audit: TopologicalAuditRecord | None = None

    @property
    def latest_audit(self) -> TopologicalAuditRecord | None:
        """Access latest topological audit record for explainability and telemetry."""
        if hasattr(self._engine, "latest_audit") and self._engine.latest_audit is not None:
            return self._engine.latest_audit
        return self._latest_audit

    def vote(self, trajectory: Trajectory) -> float:
        """Alias for evaluate_trajectory to support alternative orchestrator signatures."""
        return self.evaluate_trajectory(trajectory)

    def evaluate_trajectory(self, trajectory: Trajectory) -> float:
        """Evaluate candidate trajectory and enforce critical topological veto if corrupted.

        Returns:
            Psi_takens in [0.0, 1.0]. If audit.is_manifold_veto is True, forces 0.0 to
            trigger the critical hard-gating veto in MultiplicativeMoEGating.
        """
        try:
            if hasattr(self._engine, "evaluate_trajectory"):
                psi, audit = self._engine.evaluate_trajectory(trajectory)
                self._latest_audit = audit
                if audit.is_manifold_veto:
                    return 0.0
                return float(np.clip(psi, 0.0, 1.0))

            score = super().evaluate_trajectory(trajectory)
            if hasattr(self._engine, "latest_audit"):
                self._latest_audit = getattr(self._engine, "latest_audit", None)
                if self._latest_audit and self._latest_audit.is_manifold_veto:
                    return 0.0
            return score
        except Exception:
            return 0.0
