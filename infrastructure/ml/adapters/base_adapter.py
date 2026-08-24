"""Base adapter class for MoE engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional
import numpy as np

from infrastructure.ml.interfaces import PredictionEngine, PredictionResult
from core.orchestration.rosa_roja.domain.trajectory import Trajectory
from core.orchestration.rosa_roja.ports.expert_jury import ExpertJuryPort


class BaseExpertAdapter(ExpertJuryPort):
    """
    Base adapter wrapping a PredictionEngine as ExpertJuryPort.
    
    Does NOT modify the underlying engine - implements the protocol
    by translating trajectory evaluation to engine predictions.
    
    Handles multidimensional state vectors via PCA projection to 1D
    to preserve maximum variance information for expert evaluation.
    """
    
    def __init__(
        self,
        engine: PredictionEngine,
        name: str,
        is_critical: bool = False,
        threshold: float = 0.6,
        weight: float = 1.0,
    ):
        self._engine = engine
        self.name = name
        self.is_critical = is_critical
        self.threshold = threshold
        self.weight = weight
    
    @staticmethod
    def _project_to_1d(state_matrix: np.ndarray) -> np.ndarray:
        """
        Project multidimensional state matrix to 1D using First Principal Component.
        
        Preserves maximum variance information from multidimensional state
        (any features: price/volume, temp/pressure, etc.) instead of 
        naively taking only the first dimension.
        
        Args:
            state_matrix: Shape (T, D) where T=time steps, D=features
            
        Returns:
            1D array of shape (T,) projected onto first principal component
        """
        if state_matrix.ndim == 1:
            return state_matrix
        
        T, D = state_matrix.shape
        if D == 1:
            return state_matrix.flatten()
        
        # Center the data
        centered = state_matrix - np.mean(state_matrix, axis=0)
        
        # Check for degenerate covariance
        cov = np.cov(centered, rowvar=False)
        if cov.ndim < 2 or np.allclose(cov, 0):
            # Fallback: return primary feature (index 0)
            return state_matrix[:, 0]
        
        # Compute eigendecomposition
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            # First principal component = eigenvector with largest eigenvalue
            first_pc = eigvecs[:, np.argmax(eigvals)]
            # Project onto PC1
            projected = np.dot(centered, first_pc)
            return projected
        except np.linalg.LinAlgError:
            # Numerical issues - fallback to primary feature
            return state_matrix[:, 0]
    
    def _trajectory_to_values(self, trajectory: Trajectory) -> np.ndarray:
        """
        Extract 1D value sequence from trajectory for engine prediction.
        
        Uses PCA projection for multidimensional state to preserve
        maximum information from all state dimensions.
        """
        state_matrix = trajectory.delta_states  # Shape: (T, D)
        return self._project_to_1d(state_matrix)
    
    def evaluate_trajectory(self, trajectory: Trajectory) -> float:
        """
        Evaluate trajectory by running engine prediction on its state sequence.
        
        Returns Ψ_e(T) ∈ [0.0, 1.0] - the engine's confidence as coherence score.
        """
        try:
            values = self._trajectory_to_values(trajectory)
            result: PredictionResult = self._engine.predict(values.tolist())
            return float(np.clip(result.confidence, 0.0, 1.0))
        except Exception:
            # On any error, return low confidence rather than crashing
            return 0.0
    
    def update_learning(self, actual: float, predicted: float) -> None:
        """Propagate actual outcome to underlying engine for online learning."""
        if hasattr(self._engine, 'record_actual'):
            try:
                self._engine.record_actual(predicted, actual)
            except Exception:
                pass  # Silently ignore learning errors

    # ------------------------------------------------------------------
    # Persistence (StatePersistable contract) — delegates to the engine
    # when it supports the contract. Engines without persistence export
    # None so snapshots stay forward-compatible.
    # ------------------------------------------------------------------

    def export_state(self) -> dict:
        state = {"schema_version": 1}
        if hasattr(self._engine, "export_state"):
            state["engine"] = self._engine.export_state()
        else:
            state["engine"] = None
        return state

    def import_state(self, payload: dict) -> None:
        if not isinstance(payload, dict) or "schema_version" not in payload:
            raise ValueError(f"{self.name} adapter payload missing schema_version")
        engine_state = payload.get("engine")
        if engine_state is None:
            return
        if hasattr(self._engine, "import_state"):
            self._engine.import_state(engine_state)
        else:
            raise ValueError(
                f"Snapshot contains state for engine '{self.name}' which does "
                "not support persistence"
            )