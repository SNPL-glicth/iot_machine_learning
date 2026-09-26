"""Port specification for continuous Riemannian Manifold Engine (ZENIN v2.3+).

Conforms to:
- Hexagonal Architecture (Domain Port).
- ISO/IEC 22989: Transparent mathematical state transition contract.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from domain.entities.manifold.manifold_audit import ManifoldAuditRecord


class ManifoldEnginePort(ABC):
    """Abstract inbound/outbound domain boundary for manifold coordinate evaluation.

    Encapsulates the 3D vector field, Jacobian tensor, divergence compass,
    and 4D Ramanujan projection behind an agnostic port interface.
    """

    @abstractmethod
    def step(
        self,
        mahalanobis_d: float,
        kuramoto_r: float,
        bayesian_p: float,
        delta_time: float = 0.01,
    ) -> ManifoldAuditRecord:
        """Execute single cycle of manifold coordinate evaluation.

        Args:
            mahalanobis_d: Spatial metric distance d_M ∈ [0, ∞).
            kuramoto_r: Kuramoto phase resonance order parameter r ∈ [0, 1].
            bayesian_p: Posterior epistemic belief confidence P ∈ [0, 1].
            delta_time: Elapsed time interval Δt > 0 in seconds.

        Returns:
            ManifoldAuditRecord containing verified invariants and diagnostics.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset temporal registers and prior Jacobian history."""
        ...
