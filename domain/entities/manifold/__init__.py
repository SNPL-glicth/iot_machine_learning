"""Manifold Domain Entities and Value Objects (ZENIN v2.3+).

Exposes immutable representations for:
- 3D Continuous Riemannian State (ManifoldState3D).
- 4D Extrinsic Regularized State (ManifoldState4D).
- ISO/IEC 22989 Audit Traceability Record (ManifoldAuditRecord).
- Domain Boundary Limits (ManifoldBoundaryLimits).
"""

from __future__ import annotations

from .state_3d import ManifoldState3D
from .state_4d import ManifoldState4D
from .manifold_audit import ManifoldAuditRecord
from .manifold_parameters import ManifoldBoundaryLimits

__all__ = [
    "ManifoldState3D",
    "ManifoldState4D",
    "ManifoldAuditRecord",
    "ManifoldBoundaryLimits",
]
