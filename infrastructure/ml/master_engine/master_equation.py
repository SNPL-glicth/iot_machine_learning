"""Master Equation and continuous wave resonance module (ZENIN v2.4+).

Couples Dual Engines: Rosa Roja (Positive Pole z1) and MRT (Negative Pole z2)
via Hopf Fibration: C_Sovereign = |z1|² - |z2|² + 2|z1z2|cos(θ1 - θ2).
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
    sovereign_certainty: float = 0.0
    sovereign_polarity: float = 1.0
    delta_y_sovereign: float = 0.0
    hopf_spinor: Any | None = None


def compute_certeza(
    i_cvar: float, ds_dt: float, dr_dt: float, certeza_epistemica: float, sigma_dr: float = 1.0, epsilon: float = 1e-6,
) -> float:
    """Calculates continuous wave resonance certainty Φ_certeza ∈ [0.0, 1.0]."""
    a_risk = max(0.0, min(1.0, float(i_cvar)))
    if a_risk <= 0.0:
        return 0.0
    den = max(abs(float(dr_dt)), float(epsilon) * max(EPSILON.CONFIDENCE, abs(float(sigma_dr))))
    lam = max(0.0, min(1.0, math.exp(-min(10.0, abs(abs(float(ds_dt)) / den - 1.0)))))
    ep = max(0.0, min(1.0, float(certeza_epistemica)))
    v_s, v_r = float(ds_dt), float(dr_dt)
    if abs(v_s - v_r) < 1e-7 or abs(v_s) < 1e-9:
        k_r = 1.0
    else:
        t2, t3 = math.atan2(v_s - v_r, den), math.atan2(v_s, max(1e-4, abs(ep - 0.5)))
        k_r = min(1.0, math.hypot((1.0 + math.cos(t2) + math.cos(t3)) / 3.0, (math.sin(t2) + math.sin(t3)) / 3.0))
    return float(max(0.0, min(1.0, a_risk * lam * ep * (k_r * k_r))))


def compute_magnitud_objetivo(
    predicciones: Sequence[float] | np.ndarray, pesos: Sequence[float] | np.ndarray | None = None, default: float = 0.0,
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
    tot = float(np.sum(weights_arr))
    return float(default) if tot <= 1e-12 else float(np.sum(weights_arr * preds_arr) / tot)


def compute_momentum_veto(
    ds_dt_ema: float, magnitud: float, tau_mom: float = 0.5, sigma_mom: float = 0.001, sigma_market: float | None = None,
) -> float:
    """Calculates continuous momentum confirmation score in [0.0, 1.0]."""
    eff = max(float(sigma_mom), float(sigma_market)) if (sigma_market and sigma_market > 0) else float(sigma_mom)
    band = float(tau_mom) * eff
    net = (float(ds_dt_ema) * float(magnitud)) - band
    return 0.0 if (net <= 0.0 or band <= 0.0) else float(max(0.0, min(1.0, net / band)))


def compute_master_equation(
    phi_moe_base: float, i_cvar: float, lambda_t_crono: float,
    kuramoto_r: float = 1.0, phase_alignment: float = 1.0,
    manifold_engine: ManifoldEnginePort | None = None, delta_time: float = 0.01,
    mahalanobis_d: float | None = None, manifold_shadow_mode: bool = True,
    rosa_roja_output: Any | None = None, mrt_output: Any | None = None,
    current_reference_price: float | None = None, dual_engine_shadow_mode: bool = True,
) -> MasterEquationComponents:
    """Continuous composite evaluator uniting Rosa Roja (z1) and MRT (z2) via Hopf Fibration."""
    clamped_phi = max(0.0, min(1.0, float(phi_moe_base)))
    risk_factor = max(0.0, min(1.0, float(i_cvar)))
    clamped_lambda = max(0.0, min(1.0, float(lambda_t_crono)))
    r, align = max(0.0, min(1.0, float(kuramoto_r))), max(0.0, min(1.0, float(phase_alignment)))

    nominal_certeza = float(max(0.0, min(1.0, risk_factor * clamped_lambda * clamped_phi * (r * align))))
    final_certeza = nominal_certeza
    audit, div_val, is_4d, obj_mag = None, 0.0, False, 0.0
    sov_c, sov_pol, dy_sov, hopf_st = nominal_certeza, 1.0, 0.0, None
    geo_shadow: dict[str, Any] = {}

    # 1. Direct Dual-Engine Hopf Coupling (Rosa Roja z1 + MRT z2)
    if rosa_roja_output is not None and mrt_output is not None:
        m1 = getattr(rosa_roja_output, "metadata", {}) or getattr(rosa_roja_output, "veto_details", {}) or {}
        m2 = getattr(mrt_output, "metadata", {}) or {}
        z1_raw = float(getattr(rosa_roja_output, "confidence", getattr(rosa_roja_output, "global_confidence", nominal_certeza)))
        z1_a = float(max(0.0, min(1.0, z1_raw * risk_factor * clamped_lambda * r * align))) if risk_factor < 1.0 or clamped_lambda < 1.0 else z1_raw
        z2_a = float(m2.get("z2_amplitude", getattr(mrt_output, "confidence", 0.0)))
        env = getattr(rosa_roja_output, "envelope", None)
        y1 = float(getattr(rosa_roja_output, "predicted_value", getattr(rosa_roja_output, "magnitud_objetivo", getattr(rosa_roja_output, "prediction", getattr(env, "magnitude", 0.0)))))
        y2 = float(getattr(mrt_output, "predicted_value", 0.0))
        t1, t2 = float(m1.get("theta1_phase", 0.0)), float(m2.get("theta2_phase", 0.0))

        s0, s3 = (z1_a**2 + z2_a**2), (z1_a**2 - z2_a**2)
        s1 = 2.0 * z1_a * z2_a * math.cos(t1 - t2)
        sov_c = s3 + s1
        sov_pol = 1.0 if sov_c >= 0.0 else -1.0
        ref_p = float(current_reference_price) if (current_reference_price is not None and current_reference_price > 0) else (y1 if y1 > 0 else 1.0)
        dy_blend = ((z1_a**2 * (y1 - ref_p)) + (z2_a**2 * (y2 - ref_p))) / max(1e-12, s0)
        dy_sov = dy_blend * sov_pol
        calc_obj_mag = max(1e-6, ref_p + dy_sov)
        is_4d = True

        geo_shadow = {
            "dual_engine_mode": True, "dual_engine_shadow_mode": dual_engine_shadow_mode,
            "z1_amplitude": z1_a, "z2_amplitude": z2_a, "nominal_certeza": nominal_certeza,
            "stokes_s0": s0, "stokes_s1": s1, "stokes_s3": s3,
            "sovereign_certainty": sov_c, "sovereign_polarity": sov_pol,
            "delta_y_sovereign": dy_sov, "protected_target_magnitude": calc_obj_mag,
            "is_4d_projected": True,
        }
        if not dual_engine_shadow_mode:
            final_certeza = float(max(0.0, min(1.0, abs(sov_c))))
            obj_mag = calc_obj_mag
    # 2. Geometric Manifold Adapter Flow
    elif manifold_engine is not None:
        try:
            d_val = float(mahalanobis_d) if mahalanobis_d is not None else (1.0 - risk_factor) * 2.0
            audit = manifold_engine.step(
                mahalanobis_d=d_val, kuramoto_r=r, bayesian_p=clamped_phi, delta_time=float(delta_time),
                nominal_certainty=nominal_certeza,
            )
            div_val, is_4d, hopf_st = float(audit.divergence), audit.is_4d_projected, audit.hopf_spinor
            if hopf_st is not None:
                sov_c = float(hopf_st.sovereign_certainty)
                sov_pol = float(hopf_st.polarity_direction)
            suppressed = float(max(0.0, min(1.0, nominal_certeza * math.exp(-max(0.0, div_val)))))
            effective_c = float(hopf_st.effective_certainty) if (hopf_st is not None and is_4d) else suppressed
            geo_shadow = {
                "manifold_shadow_mode": manifold_shadow_mode,
                "nominal_certeza": nominal_certeza, "suppressed_certeza": suppressed,
                "suppression_delta": float(nominal_certeza - suppressed),
                "divergence": div_val, "is_4d_projected": is_4d,
                "sovereign_certainty": sov_c, "sovereign_polarity": sov_pol,
            }
            if not manifold_shadow_mode:
                final_certeza = effective_c
                if is_4d and audit.state_4d is not None:
                    obj_mag = float(audit.state_4d.mahalanobis_d)
        except Exception:
            pass

    return MasterEquationComponents(
        phi_moe_base=clamped_phi, i_cvar=risk_factor, lambda_t_crono=clamped_lambda,
        phi_redrose=final_certeza, certeza=final_certeza, magnitud_objetivo=obj_mag,
        momentum_veto=1.0, kuramoto_r=r, phase_alignment=align, manifold_audit=audit,
        divergence=div_val, is_4d_projected=is_4d, geometric_manifold_shadow=geo_shadow if geo_shadow else None,
        sovereign_certainty=sov_c, sovereign_polarity=sov_pol, delta_y_sovereign=dy_sov, hopf_spinor=hopf_st,
    )
