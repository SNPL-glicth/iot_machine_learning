"""Master Equation and continuous wave resonance module.

Computes continuous dimensionless resonance certainty, state manifold target
magnitude, and smooth momentum confirmation with Kuramoto phase coherence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import numpy as np
from core.parameters.numerical_constants import EPSILON


@dataclass(frozen=True)
class MasterEquationComponents:
    """Audit decomposition of terms in the Master Equation for ISO 22989 compliance."""

    phi_moe_base: float
    i_cvar: float
    lambda_t_crono: float
    phi_redrose: float
    certeza: float = 0.0
    magnitud_objetivo: float = 0.0
    momentum_veto: float = 1.0
    kuramoto_r: float = 1.0
    phase_alignment: float = 1.0


def compute_certeza(
    i_cvar: float,
    ds_dt: float,
    dr_dt: float,
    certeza_epistemica: float,
    sigma_dr: float = 1.0,
    epsilon: float = 1e-6,
) -> float:
    """Calculates continuous wave resonance certainty Φ_certeza ∈ [0.0, 1.0] with Kuramoto coupling.

    Mathematical framework:
        Interfering complex wave amplitudes:
            A1 = a_risk = clip(i_cvar, 0, 1)
            A2 = lambda_crono = exp(-min(10.0, |(ds_dt / dr_dt) - 1.0|))
            A3 = certeza_epistemica = clip(certeza_epistemica, 0, 1)
        Local Phase-Space mapping:
            theta_k = atan2(velocity, displacement_wrt_centroid)
            Stationary protection: theta1 = theta2 = theta3 = 0 when dynamics are stationary / coherent (ds_dt ≈ dr_dt).
        Kuramoto Order Parameter:
            r(t) exp(i ψ) = (1/N) Σ exp(i θ_k)
        Resonant Interference Certainty:
            |S(t)| = A1 · A2 · A3 · r(t)^2 (or phase modulation)
    """
    a_risk = max(0.0, min(1.0, float(i_cvar)))
    if a_risk <= 0.0:
        return 0.0

    effective_sigma = max(EPSILON.CONFIDENCE, abs(float(sigma_dr)))
    floor_threshold = float(epsilon) * effective_sigma
    denominator = max(abs(float(dr_dt)), floor_threshold)

    ratio = abs(float(ds_dt)) / denominator
    discrepancy = abs(ratio - 1.0)
    lambda_crono = max(0.0, min(1.0, math.exp(-min(10.0, discrepancy))))

    epistemic_clamped = max(0.0, min(1.0, float(certeza_epistemica)))
    v_s, v_r = float(ds_dt), float(dr_dt)
    delta_v = v_s - v_r

    # Stationary/synchronized check: preserve base product when no dynamic phase divergence
    if abs(delta_v) < 1e-7 or abs(v_s) < 1e-9:
        kuramoto_r = 1.0
    else:
        # Phase space coordinates: theta_k = atan2(velocity, displacement)
        theta_1 = 0.0
        theta_2 = math.atan2(delta_v, denominator)
        theta_3 = math.atan2(v_s, max(1e-4, abs(epistemic_clamped - 0.5)))
        re_z = (math.cos(theta_1) + math.cos(theta_2) + math.cos(theta_3)) / 3.0
        im_z = (math.sin(theta_1) + math.sin(theta_2) + math.sin(theta_3)) / 3.0
        kuramoto_r = min(1.0, math.hypot(re_z, im_z))

    psi_r = kuramoto_r * kuramoto_r
    certeza = a_risk * lambda_crono * epistemic_clamped * psi_r
    return float(max(0.0, min(1.0, certeza)))


def compute_magnitud_objetivo(
    predicciones: Sequence[float] | np.ndarray,
    pesos: Sequence[float] | np.ndarray | None = None,
    default: float = 0.0,
) -> float:
    """Calculates weighted consensus target magnitude in state units: μ = (Σ w_i · y_i) / (Σ w_i)."""
    preds_arr = np.asarray(predicciones, dtype=np.float64).flatten()
    if preds_arr.size == 0:
        return float(default)

    if pesos is None:
        return float(np.mean(preds_arr))

    weights_arr = np.asarray(pesos, dtype=np.float64).flatten()
    if weights_arr.size != preds_arr.size:
        return float(np.mean(preds_arr))

    total_weight = float(np.sum(weights_arr))
    if total_weight <= 1e-12:
        return float(default)

    return float(np.sum(weights_arr * preds_arr) / total_weight)


def compute_momentum_veto(
    ds_dt_ema: float,
    magnitud: float,
    tau_mom: float = 0.5,
    sigma_mom: float = 0.001,
    sigma_market: float | None = None,
) -> float:
    """Calculates continuous momentum confirmation score in [0.0, 1.0]."""
    effective_sigma = (
        max(float(sigma_mom), float(sigma_market))
        if (sigma_market is not None and sigma_market > 0)
        else float(sigma_mom)
    )
    deadband = float(tau_mom) * effective_sigma
    kinetic_flux = float(ds_dt_ema) * float(magnitud)
    net_signal = kinetic_flux - deadband

    if net_signal <= 0.0 or deadband <= 0.0:
        return 0.0

    normalized_score = net_signal / deadband
    return float(max(0.0, min(1.0, normalized_score)))


def compute_master_equation(
    phi_moe_base: float,
    i_cvar: float,
    lambda_t_crono: float,
    kuramoto_r: float = 1.0,
    phase_alignment: float = 1.0,
) -> MasterEquationComponents:
    """Continuous composite evaluator for the Master Equation with Kuramoto coherence."""
    clamped_phi_base = max(0.0, min(1.0, float(phi_moe_base)))
    risk_factor = max(0.0, min(1.0, float(i_cvar)))
    clamped_lambda = max(0.0, min(1.0, float(lambda_t_crono)))
    r = max(0.0, min(1.0, float(kuramoto_r)))
    align = max(0.0, min(1.0, float(phase_alignment)))

    resonance = risk_factor * clamped_lambda * clamped_phi_base * (r * align)
    final_certeza = float(max(0.0, min(1.0, resonance)))

    return MasterEquationComponents(
        phi_moe_base=clamped_phi_base,
        i_cvar=risk_factor,
        lambda_t_crono=clamped_lambda,
        phi_redrose=final_certeza,
        certeza=final_certeza,
        magnitud_objetivo=0.0,
        momentum_veto=1.0,
        kuramoto_r=r,
        phase_alignment=align,
    )
