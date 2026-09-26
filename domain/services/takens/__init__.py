"""Pure Mathematical Domain Services for Takens Topological Phase Space Inference.

Exposes:
- Delay coordinate extraction (extract_embedded_vector, extract_delay_matrix)
- Spectral participation dimension (compute_delay_covariance_matrix, compute_effective_dimension, compute_spectral_entropy)
- Topological fidelity and manifold veto (compute_fast_fnn_ratio, evaluate_topological_fidelity)
"""

from __future__ import annotations

from .delay_embedding_service import (
    extract_embedded_vector,
    extract_delay_matrix,
)
from .spectral_dimension_service import (
    compute_delay_covariance_matrix,
    compute_effective_dimension,
    compute_spectral_entropy,
)
from .topological_fidelity_service import (
    compute_fast_fnn_ratio,
    evaluate_topological_fidelity,
)

__all__ = [
    "extract_embedded_vector",
    "extract_delay_matrix",
    "compute_delay_covariance_matrix",
    "compute_effective_dimension",
    "compute_spectral_entropy",
    "compute_fast_fnn_ratio",
    "evaluate_topological_fidelity",
]
