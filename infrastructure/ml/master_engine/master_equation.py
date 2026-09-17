"""Master Equation pure computation module (ZENIN v2.2 specification).

Resolves dimensional errors by isolating pure dimensionless certainty,
market-unit target magnitude, and deadband momentum confirmation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class MasterEquationComponents:
    """Audit decomposition of terms in the Master Equation for ISO 22989 compliance.

    Attributes:
        phi_moe_base: Base epistemic confidence emitted by internal jury [0.0, 1.0].
        i_cvar: Binary risk veto indicator (1.0 safe, 0.0 capitulation veto).
        lambda_t_crono: Fractal chronometric synchrony index Λ(t) in [0.0, 1.0].
        phi_redrose: Final orchestrated certainty score in [0.0, 1.0].
        certeza: Alias for orchestrated certainty score in [0.0, 1.0].
        magnitud_objetivo: Consensus target magnitude in market units.
        momentum_veto: Heaviside momentum indicator (1.0 pass, 0.0 deadband veto).
    """

    phi_moe_base: float
    i_cvar: float
    lambda_t_crono: float
    phi_redrose: float
    certeza: float = 0.0
    magnitud_objetivo: float = 0.0
    momentum_veto: float = 1.0


def compute_certeza(
    i_cvar: float,
    ds_dt: float,
    dr_dt: float,
    certeza_epistemica: float,
    sigma_dr: float = 1.0,
    epsilon: float = 1e-6,
) -> float:
    """Calculates dimensionless certainty Φ_certeza ∈ [0.0, 1.0].

    Formula:
        Φ_certeza = I(CVaR_t ≤ L_max) · Λ(t) · Φ_epistémica
        where Λ(t) = exp(-|(|∂S/∂t| / max(|∂R/∂t|, ε · σ_dr)) - 1.0|)

    Args:
        i_cvar: Binary risk veto indicator [dimensionless: 1.0 safe, 0.0 veto].
        ds_dt: Price/state velocity |∂S/∂t| [fractional return/sec or price/sec].
        dr_dt: Reference rhythm velocity |∂R/∂t| [rhythm units/sec].
        certeza_epistemica: Epistemic MoE jury confidence in [0.0, 1.0] [dimensionless].
        sigma_dr: Dispersion/volatility of the reference rhythm [rhythm units/sec].
        epsilon: Numerical precision floor multiplier [dimensionless].

    Returns:
        float: Bounded orchestrated certainty score in [0.0, 1.0] [dimensionless].
    """
    # 1. Hard risk veto gate
    veto_factor = 1.0 if float(i_cvar) >= 0.5 else 0.0
    if veto_factor == 0.0:
        return 0.0

    # 2. Chronometric synchrony with zero-division protection: max(|dR/dt|, eps * sigma_dr)
    effective_sigma_dr = max(1e-6, float(abs(sigma_dr)))
    floor_threshold = float(epsilon) * effective_sigma_dr
    denominator = max(abs(float(dr_dt)), floor_threshold)

    ratio = abs(float(ds_dt)) / denominator
    discrepancy = abs(ratio - 1.0)
    # Bound discrepancy exponent to prevent numerical underflow
    lambda_crono = math.exp(-min(10.0, discrepancy))
    lambda_crono = max(0.0, min(1.0, float(lambda_crono)))

    # 3. Epistemic certainty clamping
    epistemic_clamped = max(0.0, min(1.0, float(certeza_epistemica)))

    # Multiplicative combination
    certeza = veto_factor * lambda_crono * epistemic_clamped
    return float(max(0.0, min(1.0, certeza)))


def compute_magnitud_objetivo(
    predicciones: Sequence[float] | np.ndarray,
    pesos: Sequence[float] | np.ndarray | None = None,
    default: float = 0.0,
) -> float:
    """Calculates weighted consensus target magnitude in market units.

    Formula:
        μ_target = (Σ w_i · y_i) / (Σ w_i)

    Args:
        predicciones: Vector of expert forecast magnitudes y_i [market units: return or price].
        pesos: Optional reliability weights w_i ≥ 0 [dimensionless].
        default: Fallback magnitude when input array is empty or weights sum to zero.

    Returns:
        float: Weighted average target magnitude in market units.
    """
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

    weighted_mean = float(np.sum(weights_arr * preds_arr) / total_weight)
    return weighted_mean


def compute_momentum_veto(
    ds_dt_ema: float,
    magnitud: float,
    tau_mom: float = 0.5,
    sigma_mom: float = 0.001,
) -> float:
    """Calculates Heaviside momentum confirmation with noise deadband.

    Formula:
        H((ds_dt_ema · magnitud) - (τ_mom · σ_mom))
        Returns 1.0 if signal > 0.0 (momentum confirms trade), else 0.0 (veto).

    Args:
        ds_dt_ema: Exponential moving average of price velocity [return/sec].
        magnitud: Directional target magnitude [market return units].
        tau_mom: Deadband sensitivity multiplier [dimensionless].
        sigma_mom: Noise standard deviation of momentum [return/sec].

    Returns:
        float: 1.0 if momentum confirms direction exceeding deadband, else 0.0.
    """
    signal = (float(ds_dt_ema) * float(magnitud)) - (float(tau_mom) * float(sigma_mom))
    return 1.0 if signal > 0.0 else 0.0


def compute_master_equation(
    phi_moe_base: float,
    i_cvar: float,
    lambda_t_crono: float,
) -> MasterEquationComponents:
    """Backward-compatible composite evaluator for the Master Equation."""
    clamped_phi_base = max(0.0, min(1.0, float(phi_moe_base)))
    risk_indicator = 1.0 if float(i_cvar) >= 0.5 else 0.0
    clamped_lambda_crono = max(0.0, min(1.0, float(lambda_t_crono)))

    final_phi_redrose = risk_indicator * clamped_lambda_crono * clamped_phi_base
    final_phi_redrose = max(0.0, min(1.0, final_phi_redrose))

    return MasterEquationComponents(
        phi_moe_base=clamped_phi_base,
        i_cvar=risk_indicator,
        lambda_t_crono=clamped_lambda_crono,
        phi_redrose=final_phi_redrose,
        certeza=final_phi_redrose,
        magnitud_objetivo=0.0,
        momentum_veto=1.0,
    )
