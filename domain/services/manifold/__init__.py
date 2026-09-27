"""Pure Mathematical Services for Riemannian Manifold Navigation (ZENIN v2.3+).

Exposes analytical differential operators:
- Continuous Vector Field (compute_vector_field, compute_vector_field_batch)
- Exact Jacobian Tensor (compute_jacobian_tensor, compute_jacobian_batch)
- Liouville Divergence Compass (compute_divergence, classify_manifold_regime)
- Ramanujan 4D Geodesic Regularization (execute_ramanujan_jump, propagate_geodesic_step)
"""

from __future__ import annotations

from .vector_field import (
    VectorFieldConfig,
    compute_vector_field,
    compute_vector_field_batch,
    integrate_rk4_step,
)
from .jacobian_tensor import (
    compute_jacobian_tensor,
    compute_jacobian_batch,
    verify_jacobian_numerical,
)
from .divergence_compass import (
    compute_divergence,
    compute_divergence_batch,
    compute_spectral_stability,
    classify_manifold_regime,
    evaluate_volume_ratio,
)
from .ramanujan_projection import (
    compute_metric_deformation_velocity,
    build_augmented_4d_jacobian,
    propagate_geodesic_step,
    execute_ramanujan_jump,
)
from .mrt_phase_transport import (
    reduce_angle_canonical,
    transport_phase_step,
)
from .mrt_hopf_fibration import (
    compute_metric_deformation_rate,
    compute_vorticity_curl_norm,
    compute_rational_hopf_amplitudes,
    evaluate_hopf_spinor,
)

__all__ = [
    "VectorFieldConfig",
    "compute_vector_field",
    "compute_vector_field_batch",
    "integrate_rk4_step",
    "compute_jacobian_tensor",
    "compute_jacobian_batch",
    "verify_jacobian_numerical",
    "compute_divergence",
    "compute_divergence_batch",
    "compute_spectral_stability",
    "classify_manifold_regime",
    "evaluate_volume_ratio",
    "compute_metric_deformation_velocity",
    "build_augmented_4d_jacobian",
    "propagate_geodesic_step",
    "execute_ramanujan_jump",
    "reduce_angle_canonical",
    "transport_phase_step",
    "compute_metric_deformation_rate",
    "compute_vorticity_curl_norm",
    "compute_rational_hopf_amplitudes",
    "evaluate_hopf_spinor",
]
