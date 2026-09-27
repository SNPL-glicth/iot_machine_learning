"""Ramanujan Takens Topological Inference Engine for MoE Jury.

Conforms to:
- PredictionEngine interface contract.
- Pure Hexagonal architecture: Orchestrates domain mathematical services.
- Zero Memory Fragmentation: Integrates static TakensRingBuffer.
- ISO/IEC 25010 / 22989: Fault tolerance and AI explainability telemetry.
"""

from __future__ import annotations

from typing import Any, List, Optional
import time
import numpy as np

from domain.entities.takens.takens_parameters import TakensParameters
from domain.entities.takens.topological_audit import TopologicalAuditRecord
from domain.services.takens.delay_embedding_service import extract_delay_matrix
from domain.services.takens.spectral_dimension_service import (
    compute_delay_covariance_matrix,
    compute_effective_dimension,
)
from domain.services.takens.topological_fidelity_service import evaluate_topological_fidelity
from infrastructure.ml.interfaces import PredictionEngine, PredictionResult
from .circular_buffer import TakensRingBuffer

_DEFAULT_PARAMS = TakensParameters()


class RamanujanTakensEngine(PredictionEngine):
    """Phase-space topological inference engine reconstructing latent dynamical variables."""

    def __init__(
        self,
        params: TakensParameters | None = None,
        window_size: int = 50,
    ) -> None:
        """Initialize engine with static ring buffer and domain parameters."""
        self._params = params or _DEFAULT_PARAMS
        self._window_size = max(10, window_size)
        self._buffer = TakensRingBuffer(capacity=self._params.buffer_capacity)
        self._latest_audit: TopologicalAuditRecord | None = None

    @property
    def name(self) -> str:
        """Unique engine identifier for registry and jury deliberation."""
        return "ramanujan_takens"

    @property
    def latest_audit(self) -> TopologicalAuditRecord | None:
        """Retrieve most recent topological audit record."""
        return self._latest_audit

    def can_handle(self, series: List[float]) -> bool:
        """Check if engine has sufficient historical or provided data points."""
        return (len(series) + self._buffer.size) >= self._params.m

    def push_observation(self, delta: float) -> None:
        """Directly push real-time streaming market delta into ring buffer."""
        self._buffer.push(delta)

    def evaluate_trajectory(
        self,
        trajectory_values: Any,
        timestamp_ns: int | None = None,
    ) -> tuple[float, TopologicalAuditRecord]:
        """Directly evaluate candidate trajectory against reconstructed manifold."""
        t_ns = time.time_ns() if timestamp_ns is None else timestamp_ns

        # 1. Retrieve history from ring buffer or fallback to trajectory values
        history = self._buffer.get_flat_history()
        if history.size < self._window_size:
            if hasattr(trajectory_values, "delta_states"):
                deltas = trajectory_values.delta_states
                vals = np.cumsum(deltas[:, 0]) if deltas.ndim > 1 else np.cumsum(deltas)
            else:
                vals = np.asarray(trajectory_values, dtype=np.float64).flatten()
            if history.size > 0:
                history = np.concatenate([history, vals])
            else:
                history = vals

        # 2. Extract trajectory matrix Y in R^(W x m)
        Y = extract_delay_matrix(history, window_size=self._window_size, params=self._params)

        # 3. Compute local delay covariance Sigma_tau in O(W * m^2)
        cov = compute_delay_covariance_matrix(Y, tikhonov_reg=self._params.tikhonov_regularization)

        # 4. Compute effective participation dimension D_eff in O(m^2)
        d_eff = compute_effective_dimension(cov, m_dimension=self._params.m)

        # 5. Evaluate topological fidelity and veto conditions in O(W * m)
        psi, audit = evaluate_topological_fidelity(
            trajectory_values=trajectory_values,
            historic_matrix=Y,
            effective_dimension=d_eff,
            params=self._params,
            timestamp_ns=t_ns,
        )
        self._latest_audit = audit
        return psi, audit

    def predict(
        self,
        series: List[float],
        timestamps: Optional[List[float]] = None,
    ) -> PredictionResult:
        """Implement PredictionEngine contract returning topological confidence."""
        if series:
            self._buffer.push_batch(series)

        t_ns = (
            int(timestamps[-1] * 1e9)
            if (timestamps and len(timestamps) > 0 and np.isfinite(timestamps[-1]))
            else time.time_ns()
        )

        psi, audit = self.evaluate_trajectory(series, timestamp_ns=t_ns)

        # Local linear direction estimate
        arr = np.asarray(series, dtype=np.float64) if series else self._buffer.get_flat_history(5)
        next_val = float(arr[-1]) if arr.size > 0 else 0.0
        diff = float(arr[-1] - arr[-2]) if arr.size >= 2 else 0.0

        trend_dir: Any = "up" if diff > 1e-6 else ("down" if diff < -1e-6 else "stable")

        return PredictionResult(
            predicted_value=next_val + diff,
            confidence=psi,
            trend=trend_dir,
            metadata={
                "topological_audit": audit.to_telemetry_trace(),
                "is_manifold_veto": audit.is_manifold_veto,
                "veto_reason": audit.veto_reason,
                "d_effective": audit.d_effective,
                "fnn_ratio": audit.fnn_ratio,
                "manifold_coherence": audit.manifold_coherence,
            },
        )
