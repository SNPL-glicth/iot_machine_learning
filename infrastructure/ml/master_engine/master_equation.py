"""Master Equation and continuous wave resonance module (ZENIN v2.4+).

Couples Dual Engines (Rosa Roja z1 + MRT z2 via Hopf) and Geometric Manifold into D(t).
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
    i_admissibility: float = 1.0
    variable_destino: float = 0.0

    @property
    def D_t(self) -> float:
        return self.variable_destino


def compute_certeza(i_cvar: float, ds_dt: float, dr_dt: float, certeza_epistemica: float, sigma_dr: float = 1.0, epsilon: float = 1e-6) -> float:
    """Calculates continuous wave resonance certainty Φ_certeza ∈ [0.0, 1.0]."""
    a_risk = max(0.0, min(1.0, float(i_cvar)))
    if a_risk <= 0.0:
        return 0.0
    den = max(abs(float(dr_dt)), float(epsilon) * max(EPSILON.CONFIDENCE, abs(float(sigma_dr))))
    lam = max(0.0, min(1.0, math.exp(-min(10.0, abs(abs(float(ds_dt)) / den - 1.0)))))
    ep, v_s, v_r = max(0.0, min(1.0, float(certeza_epistemica))), float(ds_dt), float(dr_dt)
    if abs(v_s - v_r) < 1e-7 or abs(v_s) < 1e-9:
        k_r = 1.0
    else:
        t2, t3 = math.atan2(v_s - v_r, den), math.atan2(v_s, max(1e-4, abs(ep - 0.5)))
        k_r = min(1.0, math.hypot((1.0 + math.cos(t2) + math.cos(t3)) / 3.0, (math.sin(t2) + math.sin(t3)) / 3.0))
    return float(max(0.0, min(1.0, a_risk * lam * ep * (k_r * k_r))))


def compute_magnitud_objetivo(predicciones: Sequence[float] | np.ndarray, pesos: Sequence[float] | np.ndarray | None = None, default: float = 0.0) -> float:
    """Calculates weighted consensus target magnitude: μ = (Σ w_i · y_i) / (Σ w_i)."""
    preds = np.asarray(predicciones, dtype=np.float64).flatten()
    if preds.size == 0:
        return float(default)
    if pesos is None:
        return float(np.mean(preds))
    w = np.asarray(pesos, dtype=np.float64).flatten()
    return float(np.mean(preds)) if w.size != preds.size else (float(default) if np.sum(w) <= 1e-12 else float(np.sum(w * preds) / np.sum(w)))


def compute_momentum_veto(ds_dt_ema: float, magnitud: float, tau_mom: float = 0.5, sigma_mom: float = 0.001, sigma_market: float | None = None) -> float:
    """Calculates continuous momentum confirmation score in [0.0, 1.0]."""
    eff = max(float(sigma_mom), float(sigma_market)) if (sigma_market and sigma_market > 0) else float(sigma_mom)
    band, net = float(tau_mom) * eff, (float(ds_dt_ema) * float(magnitud)) - (float(tau_mom) * eff)
    return 0.0 if (net <= 0.0 or band <= 0.0) else float(max(0.0, min(1.0, net / band)))


def compute_master_equation(
    phi_moe_base: float, i_cvar: float, lambda_t_crono: float, kuramoto_r: float = 1.0, phase_alignment: float = 1.0,
    manifold_engine: ManifoldEnginePort | None = None, delta_time: float = 0.01, mahalanobis_d: float | None = None,
    manifold_shadow_mode: bool = True, rosa_roja_output: Any | None = None, mrt_output: Any | None = None,
    current_reference_price: float | None = None, dual_engine_shadow_mode: bool = True,
    i_admissibility: float | None = None, mahalanobis_threshold: float = 3.0, i_takens: float = 1.0,
) -> MasterEquationComponents:
    """Continuous composite evaluator uniting Rosa Roja, MRT, and Geometric Manifold into D(t)."""
    clamped_phi, risk_factor = max(0.0, min(1.0, float(phi_moe_base))), max(0.0, min(1.0, float(i_cvar)))
    clamped_lambda = max(0.0, min(1.0, float(lambda_t_crono)))
    r, align = max(0.0, min(1.0, float(kuramoto_r))), max(0.0, min(1.0, float(phase_alignment)))

    nominal_certeza = float(max(0.0, min(1.0, risk_factor * clamped_lambda * clamped_phi * (r * align))))
    final_certeza, obj_mag, audit, div_val, is_4d = nominal_certeza, 0.0, None, 0.0, False
    sov_c, sov_pol, dy_sov, hopf_st, geo_shadow = nominal_certeza, 1.0, 0.0, None, {}

    # 1. Direct Dual-Engine Hopf Coupling (Rosa Roja z1 + MRT z2)
    if rosa_roja_output is not None and mrt_output is not None:
        m1 = getattr(rosa_roja_output, "metadata", {}) or getattr(rosa_roja_output, "veto_details", {}) or {}
        m2 = getattr(mrt_output, "metadata", {}) or {}
        z1_raw = float(getattr(rosa_roja_output, "confidence", getattr(rosa_roja_output, "global_confidence", nominal_certeza)))
        z1_a = float(max(0.0, min(1.0, z1_raw * risk_factor * clamped_lambda * r * align))) if risk_factor < 1.0 or clamped_lambda < 1.0 else z1_raw
        z2_a = float(m2.get("z2_amplitude", getattr(mrt_output, "confidence", 0.0)))
        env = getattr(rosa_roja_output, "envelope", None)
        y1 = float(getattr(rosa_roja_output, "predicted_value", getattr(rosa_roja_output, "magnitud_objetivo", getattr(rosa_roja_output, "prediction", getattr(env, "magnitude", 0.0)))))
        y2, t1, t2 = float(getattr(mrt_output, "predicted_value", 0.0)), float(m1.get("theta1_phase", 0.0)), float(m2.get("theta2_phase", 0.0))
        s0, s3, s1 = (z1_a**2 + z2_a**2), (z1_a**2 - z2_a**2), 2.0 * z1_a * z2_a * math.cos(t1 - t2)
        sov_c, sov_pol = (s3 + s1), (1.0 if (s3 + s1) >= 0.0 else -1.0)
        ref_p = float(current_reference_price) if (current_reference_price is not None and current_reference_price > 0) else (y1 if y1 > 0 else 1.0)
        dy_sov = (((z1_a**2 * (y1 - ref_p)) + (z2_a**2 * (y2 - ref_p))) / max(1e-12, s0)) * sov_pol
        calc_obj_mag, is_4d = max(1e-6, ref_p + dy_sov), True

        geo_shadow = {
            "dual_engine_mode": True, "dual_engine_shadow_mode": dual_engine_shadow_mode,
            "z1_amplitude": z1_a, "z2_amplitude": z2_a, "nominal_certeza": nominal_certeza,
            "stokes_s0": s0, "stokes_s1": s1, "stokes_s3": s3, "sovereign_certainty": sov_c, "sovereign_polarity": sov_pol,
            "delta_y_sovereign": dy_sov, "protected_target_magnitude": calc_obj_mag, "is_4d_projected": True,
        }
        if not dual_engine_shadow_mode:
            final_certeza, obj_mag = float(max(0.0, min(1.0, abs(sov_c)))), calc_obj_mag

    # 2. Geometric Manifold Flow (Liouville Volume Divergence & 4D Geodesic)
    liouville_factor = 1.0
    if manifold_engine is not None:
        try:
            d_val = float(mahalanobis_d) if mahalanobis_d is not None else (1.0 - risk_factor) * 2.0
            audit = manifold_engine.step(mahalanobis_d=d_val, kuramoto_r=r, bayesian_p=clamped_phi, delta_time=float(delta_time), nominal_certainty=nominal_certeza)
            div_val, is_4d_man, hopf_st = float(audit.divergence), audit.is_4d_projected, audit.hopf_spinor
            if is_4d_man: is_4d = True
            if hopf_st is not None and not geo_shadow.get("dual_engine_mode", False):
                sov_c, sov_pol = float(hopf_st.sovereign_certainty), float(hopf_st.polarity_direction)

            liouville_factor = float(math.exp(-max(0.0, div_val)))
            suppressed = float(max(0.0, min(1.0, final_certeza * liouville_factor)))
            geo_shadow.update({
                "manifold_shadow_mode": manifold_shadow_mode, "nominal_certeza": nominal_certeza,
                "suppressed_certeza": suppressed, "suppression_delta": float(final_certeza - suppressed),
                "divergence": div_val, "liouville_factor": liouville_factor, "is_4d_projected": is_4d,
                "sovereign_certainty": sov_c, "sovereign_polarity": sov_pol,
            })
            if not manifold_shadow_mode:
                final_certeza = suppressed
                if is_4d_man and audit.state_4d is not None and not geo_shadow.get("dual_engine_mode", False):
                    obj_mag = float(audit.state_4d.mahalanobis_d)
        except Exception:
            pass

    # 3. Factor 1: Admissibility indicator product ∏_k I_k(t)
    i_m = 0.0 if (mahalanobis_d is not None and mahalanobis_d > mahalanobis_threshold) else 1.0
    prod_i = float(i_admissibility) if i_admissibility is not None else float(i_m * float(i_takens) * (1.0 if risk_factor > 0.0 else 0.0))

    # =========================================================================
    # D(t) Unified Destination Variable: [∏_k I_k(t)] · exp(-max(0, div F(t))) · Pi_sov(t) · |S3(t) + S1(t)|
    # Factor 1: ∏_k I_k(t) in {0, 1} -> Absolute admissibility veto (Mahalanobis, Takens, Risk)
    # Factor 2: exp(-max(0, div F(t))) -> Liouville volumetric dissipation damping on manifold M
    # Factor 3: Pi_sov(t) in {-1, +1} -> Sovereign phase polarity from Hopf fibration
    # Factor 4: |S3(t) + S1(t)| -> Sovereign coupled resonance certainty magnitude (|C_sovereign|)
    # =========================================================================
    var_destino = float(prod_i * liouville_factor * sov_pol * abs(sov_c))
    if geo_shadow:
        geo_shadow.update({"variable_destino": var_destino, "D_t": var_destino, "prod_i": prod_i})

    return MasterEquationComponents(
        phi_moe_base=clamped_phi, i_cvar=risk_factor, lambda_t_crono=clamped_lambda,
        phi_redrose=final_certeza, certeza=final_certeza, magnitud_objetivo=obj_mag,
        momentum_veto=1.0, kuramoto_r=r, phase_alignment=align, manifold_audit=audit,
        divergence=div_val, is_4d_projected=is_4d, geometric_manifold_shadow=geo_shadow if geo_shadow else None,
        sovereign_certainty=sov_c, sovereign_polarity=sov_pol, delta_y_sovereign=dy_sov, hopf_spinor=hopf_st,
        i_admissibility=prod_i, variable_destino=var_destino,
    )
