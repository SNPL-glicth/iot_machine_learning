"""Embedded phase-space state Value Object in R^m for Takens Engine.

Conforms to:
- ISO/IEC 25010: Reliability and Numerical Fault Tolerance (NaN/Inf neutralization).
- Pure Hexagonal Domain Entity: Completely decoupled from I/O and infrastructure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True, slots=True)
class EmbeddedState:
    """Immutable state vector reconstructed in phase space via delay coordinates.

    Attributes:
        delay_vector: Contiguous float64 vector y_t = [x_t, x_{t-tau}, ...] in R^m.
        effective_dimension: Participation ratio D_eff = (Tr Sigma)^2 / Tr(Sigma^2) in [1.0, m].
        fnn_ratio: False Nearest Neighbors ratio Omega_FNN in [0.0, 1.0].
        metric_energy: Kinetic/deformation energy scale E = 0.5 * ||y_t||^2.
        timestamp_ns: Discrete unix timestamp in nanoseconds for strict event sequencing.
    """

    delay_vector: np.ndarray
    effective_dimension: float
    fnn_ratio: float
    metric_energy: float = 0.0
    timestamp_ns: int = 0

    def __post_init__(self) -> None:
        """Enforce strict numerical sanitization and array immutability (ISO 25010)."""
        # 1. Sanitize delay_vector: enforce 1D float64, neutralize NaNs and Infinities
        arr = np.asarray(self.delay_vector, dtype=np.float64).flatten()
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Enforce read-only array to preserve frozen dataclass invariants
        arr.flags.writeable = False
        object.__setattr__(self, "delay_vector", arr)

        # 2. Sanitize effective_dimension
        m_dim = max(1.0, float(arr.size))
        if not np.isfinite(self.effective_dimension):
            d_eff = 1.0
        else:
            d_eff = max(1.0, min(m_dim, float(self.effective_dimension)))
        object.__setattr__(self, "effective_dimension", d_eff)

        # 3. Sanitize fnn_ratio in [0.0, 1.0]
        if not np.isfinite(self.fnn_ratio):
            fnn = 0.0
        else:
            fnn = max(0.0, min(1.0, float(self.fnn_ratio)))
        object.__setattr__(self, "fnn_ratio", fnn)

        # 4. Compute metric energy if zero or sanitize provided value
        if self.metric_energy <= 0.0 or not np.isfinite(self.metric_energy):
            energy = 0.5 * float(np.sum(arr * arr))
        else:
            energy = max(0.0, float(self.metric_energy))
        object.__setattr__(self, "metric_energy", energy)

        # 5. Sanitize timestamp
        object.__setattr__(self, "timestamp_ns", max(0, self.timestamp_ns))

    @property
    def embedding_dimension(self) -> int:
        """Return the actual embedding dimension m of the vector."""
        return self.delay_vector.size

    def to_numpy(self) -> np.ndarray:
        """Return a copy of the delay vector as a writable array."""
        return np.array(self.delay_vector, copy=True, dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        """Convert state properties to primitive dictionary."""
        return {
            "delay_vector": self.delay_vector.tolist(),
            "embedding_dimension": self.embedding_dimension,
            "effective_dimension": self.effective_dimension,
            "fnn_ratio": self.fnn_ratio,
            "metric_energy": self.metric_energy,
            "timestamp_ns": self.timestamp_ns,
        }
