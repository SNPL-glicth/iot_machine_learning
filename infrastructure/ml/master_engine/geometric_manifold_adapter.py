"""Infrastructure Adapter for Geometric Manifold Engine with Hopf Spinor (ZENIN v2.4+).

Implements ManifoldEnginePort, orchestrating:
- Exact Jacobian Tensor & Liouville Divergence invariants.
- Dissipative U(1) parallel transport of phase to neutralize aliasing.
- Rational algebraic Hopf Spinor projection onto Stokes S² manifold.
- Ramanujan 4D geodesic regularization for metric shock invariants.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from domain.entities.manifold.hopf_spinor_state import HopfSpinorState
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
from domain.services.manifold.mrt_hopf_fibration import (
    compute_metric_deformation_rate,
    compute_vorticity_curl_norm,
    evaluate_hopf_spinor,
)
from domain.services.manifold.mrt_phase_transport import transport_phase_step
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
        self._prev_phase_delta: float = 0.0
        self._timestamp: float = 0.0

    @property
    def config(self) -> VectorFieldConfig:
        return self._cfg

    def reset(self) -> None:
        """Reset temporal state history, phase memory, and prior Jacobian register."""
        self._prev_jacobian = None
        self._prev_phase_delta = 0.0
        self._timestamp = 0.0

    def step(
        self,
        mahalanobis_d: float,
        kuramoto_r: float,
        bayesian_p: float,
        delta_time: float = 0.01,
        nominal_certainty: float | None = None,
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

        # 2. Metric Deformation & Maxwell Vorticity
        frob_rate = 0.0
        is_shock = False
        if self._prev_jacobian is not None:
            _, frob_rate = compute_metric_deformation_rate(J, self._prev_jacobian, dt)
            if frob_rate > self._tau_shock:
                is_shock = True

        curl_b = compute_vorticity_curl_norm(J)

        # 3. Dissipative Parallel Transport of Phase (U(1) Bundle)
        next_phase, _, _ = transport_phase_step(
            div_e=div_val,
            curl_b_norm=curl_b,
            prev_phase_delta=self._prev_phase_delta,
            delta_time=dt,
        )
        self._prev_phase_delta = next_phase

        # 4. Rational Algebraic Hopf Spinor Projection
        base_c = float(nominal_certainty) if nominal_certainty is not None else float(kuramoto_r * bayesian_p)
        hopf_spinor: HopfSpinorState = evaluate_hopf_spinor(
            nominal_certainty=base_c,
            frob_norm=frob_rate,
            phase_delta=next_phase,
            trace_j4d_star=float(np.trace(J)) if div_val < 0.0 else -0.1,
        )

        # 5. 4D Ramanujan Geodesic Jump (Preserved for Shock Invariants)
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

        # 6. Construct Verified ISO/IEC 22989 Audit Record
        audit = ManifoldAuditRecord.create(
            timestamp=self._timestamp,
            state_3d=s3,
            jacobian_matrix=J,
            state_4d=state_4d,
            hopf_spinor=hopf_spinor,
        )

        # 7. Advance State Registers
        self._prev_jacobian = J
        self._timestamp += dt

        return audit
