"""Parameters and boundary limits for Takens Delay Embedding Engine.

Conforms to:
- ISO/IEC 25010: Reliability, Fault Tolerance, and Strict Boundary Validation.
- Pure Hexagonal Domain Entity: Zero infrastructure dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TakensParameters:
    """Immutable parameters controlling phase-space embedding and topological gating.

    Attributes:
        m: Embedding dimension m >= 2 in R^m (satisfies Takens m >= 2d + 1).
        tau_strides: Discrete delay intervals for time-lagged coordinates.
        buffer_capacity: Static circular ring-buffer size (must be power of 2).
        tau_fnn: False Nearest Neighbors threshold triggering topological veto.
        spectral_collapse_threshold: Minimum effective dimension D_eff before veto.
        tikhonov_regularization: Numerical ridge constant for Gram matrix stability.
        max_geodesic_distance: Maximum distance to manifold tangent before veto.
        out_of_manifold_penalty: Soft confidence penalty factor for off-manifold states.
    """

    m: int = 4
    tau_strides: tuple[int, ...] = (1, 2, 4, 8)
    buffer_capacity: int = 1024
    tau_fnn: float = 0.40
    spectral_collapse_threshold: float = 1.15
    tikhonov_regularization: float = 1e-6
    max_geodesic_distance: float = 3.0
    out_of_manifold_penalty: float = 0.5

    def __post_init__(self) -> None:
        """Validate invariant constraints on construction."""
        if self.m < 2:
            raise ValueError(f"Embedding dimension m must be >= 2, received {self.m}")

        if (self.buffer_capacity <= 0) or ((self.buffer_capacity & (self.buffer_capacity - 1)) != 0):
            raise ValueError(
                f"buffer_capacity must be a positive power of 2, received {self.buffer_capacity}"
            )

        if not self.tau_strides or any(s <= 0 for s in self.tau_strides):
            raise ValueError(f"tau_strides must contain strictly positive integers: {self.tau_strides}")

        if self.tau_fnn <= 0.0 or self.tau_fnn > 1.0:
            raise ValueError(f"tau_fnn threshold must be in (0.0, 1.0], received {self.tau_fnn}")

        if self.spectral_collapse_threshold < 1.0:
            raise ValueError(
                f"spectral_collapse_threshold must be >= 1.0, received {self.spectral_collapse_threshold}"
            )
