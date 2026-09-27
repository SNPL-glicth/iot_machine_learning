"""Liouville Divergence Compass and Phase-Space Volume Invariant Service.

Evaluates div(F) = Tr(J) to determine instantaneous rate of volume expansion/contraction
d(δV)/dt = Tr(J) δV, serving as early warning radar before state collapse.
"""

from __future__ import annotations

from typing import cast
import numpy as np

from domain.entities.manifold.manifold_parameters import ManifoldBoundaryLimits

_DEFAULT_LIMITS = ManifoldBoundaryLimits()


def compute_divergence(jacobian: np.ndarray) -> float:
    """Compute differential divergence div(F) = Tr(J) for a single square matrix."""
    J = np.asarray(jacobian, dtype=np.float64)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"Expected square Jacobian matrix, received shape {J.shape}")
    return float(np.trace(J))


def compute_divergence_batch(jacobians: np.ndarray) -> np.ndarray:
    """Batch evaluation of divergences using Einstein summation contraction."""
    J = np.asarray(jacobians, dtype=np.float64)
    return cast(np.ndarray, np.einsum("...ii->...", J))


def compute_spectral_stability(
    jacobian: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Calculate spectral properties and orientation preservation of the flow."""
    J = np.asarray(jacobian, dtype=np.float64)
    det_val = float(np.linalg.det(J))
    eigvals = np.linalg.eigvals(J)
    max_real_eig = float(np.max(np.real(eigvals)))
    return det_val, max_real_eig, eigvals


def classify_manifold_regime(
    divergence: float,
    max_real_eig: float,
    limits: ManifoldBoundaryLimits | None = None,
) -> tuple[bool, str]:
    """Diagnose topological stability regime from divergence and spectral radius."""
    lim = limits or _DEFAULT_LIMITS
    div_v = float(divergence)
    eig_v = float(max_real_eig)

    if div_v > lim.divergence_threshold or eig_v > lim.spectral_threshold:
        return True, "EXPANSIVE_CHAOS"

    if div_v < lim.neutral_contraction_threshold:
        return False, "CONTRACTIVE_ATTRACTOR"

    return False, "NEUTRAL_TRANSITION"


def classify_detailed_spectral_regime(
    eigvals: np.ndarray,
    divergence: float,
    limits: ManifoldBoundaryLimits | None = None,
) -> tuple[bool, str, dict[str, float]]:
    """Fine-grained spectral decomposition into chaotic, oscillatory limit cycle, or sink."""
    lim = limits or _DEFAULT_LIMITS
    reals = np.real(eigvals)
    imags = np.imag(eigvals)
    max_real = float(np.max(reals))
    max_imag = float(np.max(np.abs(imags)))
    div_v = float(divergence)

    metrics = {"max_real_eig": max_real, "max_imag_eig": max_imag, "divergence": div_v}

    if div_v > lim.divergence_threshold or max_real > lim.spectral_threshold:
        return True, "EXPANSIVE_CHAOS", metrics

    if max_imag > max(0.01, abs(max_real)):
        return False, "LIMIT_CYCLE_OSCILLATORY", metrics

    if div_v < lim.neutral_contraction_threshold:
        return False, "CONTRACTIVE_ATTRACTOR", metrics

    return False, "NEUTRAL_TRANSITION", metrics


def evaluate_volume_ratio(
    initial_volume: float,
    divergence: float,
    delta_time: float,
    limits: ManifoldBoundaryLimits | None = None,
) -> float:
    """Propagate infinitesimal volume element under Liouville flow:
        V(t + Δt) = V(t) * exp(div(F) * Δt)
    """
    lim = limits or _DEFAULT_LIMITS
    v0 = max(lim.min_initial_volume, float(initial_volume))
    dt = max(0.0, float(delta_time))
    growth_rate = np.clip(float(divergence) * dt, -lim.volume_exponent_clip, lim.volume_exponent_clip)
    return float(v0 * np.exp(growth_rate))
