"""Master Equation and continuous wave resonance module (ZENIN v2.3+).

Computes continuous dimensionless resonance certainty, state manifold target magnitude,
and smooth momentum confirmation with Kuramoto phase coherence.
Injects ManifoldEnginePort with Active Control sovereign modulation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING, Any
import numpy as np

from core.parameters.numerical_constants import EPSILON

if TYPE_CHECKING:
    from domain.ports.manifold.manifold_engine_port import ManifoldEnginePort


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
    manifold_audit: Any | None = None
    divergence: float = 0.0
    is_4d_projected: bool = False
    geometric_manifold_shadow: dict[str, Any] | None = None


def compute_certeza(
    i_cvar: float,
    ds_dt: float,
    dr_dt: float,
    certeza_epistemica: float,
    sigma_dr: float = 1.0,
    epsilon: float = 1e-6,
) -> float:
    """Calculates continuous wave resonance certainty Φ_certeza ∈ [0.0, 1.0]."""
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
    if abs(v_s - v_r) < 1e-7 or abs(v_s) < 1e-9:
        kuramoto_r = 1.0
    else:
        t2 = math.atan2(v_s - v_r, denominator)
        t3 = math.atan2(v_s, max(1e-4, abs(epistemic_clamped - 0.5)))
        kuramoto_r = min(1.0, math.hypot((1.0 + math.cos(t2) + math.cos(t3)) / 3.0, (math.sin(t2) + math.sin(t3)) / 3.0))

    certeza = a_risk * lambda_crono * epistemic_clamped * (kuramoto_r * kuramoto_r)
    return float(max(0.0, min(1.0, certeza)))


def compute_magnitud_objetivo(
    predicciones: Sequence[float] | np.ndarray,
    pesos: Sequence[float] | np.ndarray | None = None,
    default: float = 0.0,
) -> float:
    """Calculates weighted consensus target magnitude: μ = (Σ w_i · y_i) / (Σ w_i)."""
    preds_arr = np.asarray(predicciones, dtype=np.float64).flatten()
    if preds_arr.size == 0:
        return float(default)
    if pesos is None:
        return float(np.mean(preds_arr))
    weights_arr = np.asarray(pesos, dtype=np.float64).flatten()
    if weights_arr.size != preds_arr.size:
        return float(np.mean(preds_arr))
    total_weight = float(np.sum(weights_arr))
    return float(default) if total_weight <= 1e-12 else float(np.sum(weights_arr * preds_arr) / total_weight)


def compute_momentum_veto(
    ds_dt_ema: float,
    magnitud: float,
    tau_mom: float = 0.5,
    sigma_mom: float = 0.001,
    sigma_market: float | None = None,
) -> float:
    """Calculates continuous momentum confirmation score in [0.0, 1.0]."""
    eff_sig = max(float(sigma_mom), float(sigma_market)) if (sigma_market and sigma_market > 0) else float(sigma_mom)
    deadband = float(tau_mom) * eff_sig
    net_signal = (float(ds_dt_ema) * float(magnitud)) - deadband
    if net_signal <= 0.0 or deadband <= 0.0:
        return 0.0
    return float(max(0.0, min(1.0, net_signal / deadband)))


def compute_master_equation(
    phi_moe_base: float,
    i_cvar: float,
    lambda_t_crono: float,
    kuramoto_r: float = 1.0,
    phase_alignment: float = 1.0,
    manifold_engine: ManifoldEnginePort | None = None,
    delta_time: float = 0.01,
    mahalanobis_d: float | None = None,
    manifold_shadow_mode: bool = True,
) -> MasterEquationComponents:
    """Continuous composite evaluator with dedicated manifold shadow observation mode."""
    clamped_phi = max(0.0, min(1.0, float(phi_moe_base)))
    risk_factor = max(0.0, min(1.0, float(i_cvar)))
    clamped_lambda = max(0.0, min(1.0, float(lambda_t_crono)))
    r, align = max(0.0, min(1.0, float(kuramoto_r))), max(0.0, min(1.0, float(phase_alignment)))

    nominal_certeza = float(max(0.0, min(1.0, risk_factor * clamped_lambda * clamped_phi * (r * align))))
    final_certeza = nominal_certeza
    audit, div_val, is_4d, obj_mag = None, 0.0, False, 0.0
    geo_shadow: dict[str, Any] = {}

    if manifold_engine is not None:
        try:
            d_val = float(mahalanobis_d) if mahalanobis_d is not None else (1.0 - risk_factor) * 2.0
            audit = manifold_engine.step(
                mahalanobis_d=d_val, kuramoto_r=r, bayesian_p=clamped_phi, delta_time=float(delta_time)
            )
            div_val = float(audit.divergence)
            is_4d = bool(audit.is_4d_projected)
            suppressed = float(max(0.0, min(1.0, nominal_certeza * math.exp(-max(0.0, div_val)))))
            geo_shadow = {
                "manifold_shadow_mode": bool(manifold_shadow_mode),
                "nominal_certeza": nominal_certeza,
                "suppressed_certeza": suppressed,
                "suppression_delta": float(nominal_certeza - suppressed),
                "divergence": div_val,
                "is_4d_projected": is_4d,
            }
            if not manifold_shadow_mode:
                final_certeza = suppressed
                if is_4d and audit.state_4d is not None:
                    obj_mag = float(audit.state_4d.mahalanobis_d)
        except Exception:
            pass

    return MasterEquationComponents(
        phi_moe_base=clamped_phi,
        i_cvar=risk_factor,
        lambda_t_crono=clamped_lambda,
        phi_redrose=final_certeza,
        certeza=final_certeza,
        magnitud_objetivo=obj_mag,
        momentum_veto=1.0,
        kuramoto_r=r,
        phase_alignment=align,
        manifold_audit=audit,
        divergence=div_val,
        is_4d_projected=is_4d,
        geometric_manifold_shadow=geo_shadow if geo_shadow else None,
    )
