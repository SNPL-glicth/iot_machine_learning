"""Domain boundary limits and numerical parameters for the Manifold Engine (ZENIN v2.3+).

Conforms to:
- ISO/IEC 25010: Reliability and Fault Tolerance.
- Elimination of hardcoded magic numbers across the manifold domain.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ManifoldBoundaryLimits:
    """Consolidated boundary limits, clamping thresholds, and numerical parameters."""

    max_mahalanobis_clip: float = 100.0
    kuramoto_sync_threshold: float = 0.85
    frobenius_shock_threshold: float = 25.0
    divergence_clip_min: float = -10.0
    divergence_clip_max: float = 10.0
    divergence_threshold: float = 0.5
    spectral_threshold: float = 0.8
    neutral_contraction_threshold: float = -0.05
    sech_input_clip: float = 15.0
    volume_exponent_clip: float = 20.0
    lambda_4_base: float = 5.0
    lambda_4_scale: float = 2.5
    lambda_4_offset: float = 2.0
    coupling_4d_default: float = 0.15
    eps_denominator: float = 1e-6
    min_initial_volume: float = 1e-12
