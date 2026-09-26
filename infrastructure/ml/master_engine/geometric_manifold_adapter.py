"""Infrastructure Adapter for Geometric Manifold Engine (ZENIN v2.3+).

Implements ManifoldEnginePort, holding temporal state (J_{t-1}) to compute metric
derivatives and orchestrating pure domain services in Shadow/Active mode.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from domain.entities.manifold.manifold_audit import ManifoldAuditRecord
from domain.entities.manifold.state_3d import ManifoldState3D
from domain.entities.manifold.state_4d import ManifoldState4D
from domain.ports.manifold.manifold_engine_port import ManifoldEnginePort
from domain.services.manifold.divergence_compass import (
    classify_manifold_regime,
    compute_divergence,
    compute_spectral_stability,
)
from domain.services.manifold.jacobian_tensor import compute_jacobian_tensor
from domain.services.manifold.ramanujan_projection import (
    compute_metric_deformation_velocity,
    execute_ramanujan_jump,
)
from domain.services.manifold.vector_field import VectorFieldConfig


class GeometricManifoldAdapter(ManifoldEnginePort):
    """Adapter bridging the continuous manifold domain engine with master orchestration."""

    def __init__(
        self,
        config: VectorFieldConfig | None = None,
        tau_frobenius_shock: float = 15.0,
    ) -> None:
        self._cfg = config or VectorFieldConfig()
        self._tau_shock = float(tau_frobenius_shock)
        self._prev_jacobian: np.ndarray | None = None
        self._timestamp: float = 0.0

    @property
    def config(self) -> VectorFieldConfig:
        return self._cfg

    def reset(self) -> None:
        """Reset temporal state history and prior Jacobian register."""
        self._prev_jacobian = None
        self._timestamp = 0.0

    def step(
        self,
        mahalanobis_d: float,
        kuramoto_r: float,
        bayesian_p: float,
        delta_time: float = 0.01,
    ) -> ManifoldAuditRecord:
        """Execute single step of manifold flow and return verifiable audit record."""
        dt = max(1e-6, float(delta_time))
        s3 = ManifoldState3D(
            mahalanobis_d=mahalanobis_d,
            kuramoto_r=kuramoto_r,
            bayesian_p=bayesian_p,
        )

        # 1. Analytical Jacobian Tensor & Divergence Invariants
        J = compute_jacobian_tensor(s3, self._cfg)
        div_val = compute_divergence(J)
        det_val, max_real_eig, _ = compute_spectral_stability(J)
        is_unstable, regime = classify_manifold_regime(div_val, max_real_eig)

        # 2. Detect Metric Shock via Frobenius Rate
        is_shock = False
        if self._prev_jacobian is not None:
            _, frob_rate, _ = compute_metric_deformation_velocity(J, self._prev_jacobian, dt)
            if frob_rate > self._tau_shock:
                is_shock = True

        # 3. 4D Ramanujan Jump or In-Manifold Flow
        state_4d: ManifoldState4D | None = None
        if (is_unstable or is_shock) and self._prev_jacobian is not None:
            reason = "divergence_explosion" if is_unstable else "metric_shock_surge"
            state_4d, _, _ = execute_ramanujan_jump(
                state_3d=s3,
                j_curr=J,
                j_prev=self._prev_jacobian,
                delta_time=dt,
                trigger_reason=reason,
            )

        # 4. Construct Immutable ISO 22989 Audit Record
        audit = ManifoldAuditRecord.create(
            timestamp=self._timestamp,
            state_3d=s3,
            jacobian_matrix=J,
            state_4d=state_4d,
        )

        # 5. Advance State
        self._prev_jacobian = J
        self._timestamp += dt

        return audit
