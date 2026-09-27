"""Master Equation Orchestrator (Phase-Space Resonance & Wave Superposition).

Integrates Rosa Roja trajectory engine, MRT phase conjugate engine, continuous stochastic
risk adapter, fractal chronometric adapter, and Geometric Manifold Engine into an agnostic orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from core.parameters.numerical_constants import EPSILON
from domain.entities.rosa_roja.execution import ExecutionPlan
from infrastructure.ml.engines.rosa_roja.algorithms.engine import RosaRojaEngine
from infrastructure.ml.master_engine.geometric_manifold_adapter import GeometricManifoldAdapter
from infrastructure.ml.master_engine.master_equation import (
    compute_certeza,
    compute_master_equation,
    compute_momentum_veto,
)
from infrastructure.ml.master_engine.plan_builder import (
    build_orchestrated_execution_plan,
    extract_expert_target_magnitude,
)
from infrastructure.ml.master_engine.port import MasterDecisionPort
from infrastructure.ml.master_engine.telemetry import (
    assemble_master_trace,
    compute_telemetry_hash,
    extract_base_trace,
)

logger = logging.getLogger(__name__)


class MasterEquationOrchestrator(MasterDecisionPort):
    """Top-level master orchestrator executing continuous wave resonance on Manifold M."""

    def __init__(
        self, rosa_roja_engine: RosaRojaEngine, risk_adapter: Any, temporal_adapter: Any, *,
        mrt_engine: Any | None = None, manifold_engine: Any | None = None, shadow_mode: bool = False,
        manifold_shadow_mode: bool = True, dual_engine_shadow_mode: bool = True,
        tau_mom: float = 0.5, sigma_mom: float = 0.001, gamma_exec: float | None = None,
    ) -> None:
        self._rosa_roja = rosa_roja_engine
        self._mrt = mrt_engine
        self._risk_adapter = risk_adapter
        self._temporal_adapter = temporal_adapter
        self._manifold_engine = manifold_engine
        self._shadow_mode, self._manifold_shadow_mode = shadow_mode, manifold_shadow_mode
        self._dual_engine_shadow_mode = dual_engine_shadow_mode
        self._tau_mom, self._sigma_mom, self._gamma_exec = tau_mom, sigma_mom, gamma_exec

    @property
    def gamma_exec(self) -> float:
        val = self._gamma_exec if self._gamma_exec is not None else getattr(self._rosa_roja, "gamma_exec", 0.5)
        return float(val) if isinstance(val, (int, float)) else 0.5

    @property
    def mrt_engine(self) -> Any | None:
        return self._mrt

    @property
    def state_machine(self) -> Any:
        return self._rosa_roja.state_machine

    def cancel_active_trajectory(self) -> None:
        self._rosa_roja.cancel_active_trajectory()

    def reset(self) -> None:
        self._rosa_roja.reset()
        for adapter in (self._risk_adapter, self._temporal_adapter, self._manifold_engine, self._mrt):
            if adapter is not None and hasattr(adapter, "reset"):
                adapter.reset()

    def process_event(self, delta_state: np.ndarray, delta_time: float) -> ExecutionPlan:
        """Process agnostic state transition ΔS through the resonance pipeline."""
        plan_base = self._rosa_roja.process_event(delta_state, delta_time)
        base_trace = extract_base_trace(plan_base)
        phi_moe_base = float(base_trace.get("phi_moe_base", base_trace.get("phi_moe", plan_base.global_confidence)))

        arr = np.asarray(delta_state, dtype=np.float64).flatten()
        state_norm = float(np.linalg.norm(arr)) if arr.size > 0 else 0.0
        dt = max(EPSILON.DIVISION, float(delta_time))
        velocity = state_norm / dt
        state_dispersion = float(np.std(arr)) if arr.size > 1 else max(state_norm, 0.0001)

        risk_verdict: dict[str, Any] = {}
        try:
            risk_verdict = self._risk_adapter.record_observation(return_signal=state_norm, delta_time=dt, log_return=state_norm)
        except Exception as exc:
            logger.debug("risk_adapter_error: %s", exc)

        temporal_verdict: dict[str, Any] = {}
        try:
            temporal_verdict = self._temporal_adapter.record_observation(current_price=state_norm, price_velocity=velocity)
        except Exception as exc:
            logger.debug("temporal_adapter_error: %s", exc)

        i_cvar = float(risk_verdict.get("veto_riesgo", 1.0))
        lambda_t_crono = float(temporal_verdict.get("lambda_crono", 1.0))
        ds_dt = float(temporal_verdict.get("dS_dt", velocity))
        dr_dt = float(temporal_verdict.get("dR_dt", max(state_dispersion, 0.0001)))

        traj = plan_base.chosen_trajectory
        alignment = 1.0
        if traj is not None and getattr(traj, "directions", None) and len(traj.directions) > 0 and state_norm > 0:
            ref_dir = traj.directions[0]
            if ref_dir.shape == arr.shape:
                alignment = float(np.clip(np.dot(arr / state_norm, ref_dir), -1.0, 1.0))

        jury = getattr(self._rosa_roja, "_jury", [])
        expert_mag = extract_expert_target_magnitude(jury, plan_base)

        mrt_res = None
        if self._mrt is not None:
            try:
                vals = [float(m.delta_state[0]) for m in traj.movements] if (traj and traj.movements and len(traj.movements) >= 4) else arr.tolist()
                mrt_res = self._mrt.predict(vals, timestamps=[0.0, dt])
            except Exception as exc:
                logger.debug("mrt_predict_error: %s", exc)

        mahal_d = float(base_trace.get("mahalanobis_dist", base_trace.get("mahal_dist", state_norm)))
        ing = getattr(self._rosa_roja, "_ingestion", None)
        i_mahal = 0.0 if mahal_d > float(getattr(ing, "noise_threshold", 3.0)) else 1.0
        i_takens = 1.0
        for exp in jury:
            if getattr(exp, "name", "") == "ramanujan_takens" or hasattr(exp, "latest_audit"):
                aud = getattr(exp, "latest_audit", None)
                if aud is not None and getattr(aud, "is_manifold_veto", False):
                    i_takens = 0.0
                    break
        v_det = getattr(plan_base, "veto_details", None)
        if isinstance(v_det, dict) and ("fnn" in str(v_det).lower() or v_det.get("expert_name") == "ramanujan_takens"):
            i_takens = 0.0
        prod_i = float(i_mahal * i_takens * (1.0 if i_cvar > 0.0 else 0.0))

        comp = compute_master_equation(
            phi_moe_base=phi_moe_base, i_cvar=i_cvar, lambda_t_crono=lambda_t_crono,
            kuramoto_r=float(base_trace.get("kuramoto_r", 1.0)), phase_alignment=alignment,
            manifold_engine=self._manifold_engine, delta_time=dt, mahalanobis_d=mahal_d,
            manifold_shadow_mode=self._manifold_shadow_mode,
            dual_engine_shadow_mode=self._dual_engine_shadow_mode,
            rosa_roja_output=plan_base, mrt_output=mrt_res,
            current_reference_price=expert_mag if expert_mag > 0 else state_norm,
            i_admissibility=prod_i,
        )

        eff_sigma_dr = max(0.1, state_dispersion * 2.0)
        c_legacy = compute_certeza(i_cvar, ds_dt, dr_dt, phi_moe_base, eff_sigma_dr)
        certeza = c_legacy if self._shadow_mode else comp.certeza

        magnitud_objetivo = expert_mag
        if not self._shadow_mode and comp.is_4d_projected and comp.magnitud_objetivo > 0.0:
            if (not self._manifold_shadow_mode and comp.manifold_audit is not None) or (mrt_res is not None and not self._dual_engine_shadow_mode):
                magnitud_objetivo = comp.magnitud_objetivo

        momentum_veto = compute_momentum_veto(
            ds_dt_ema=float(alignment * ds_dt), magnitud=magnitud_objetivo,
            tau_mom=self._tau_mom, sigma_mom=self._sigma_mom, sigma_market=state_dispersion,
        )

        telemetry_hash = compute_telemetry_hash(delta_state, delta_time)
        master_trace = assemble_master_trace(
            base_trace=base_trace, phi_moe_base=phi_moe_base, i_cvar=i_cvar, lambda_t_crono=lambda_t_crono,
            certeza=certeza, magnitud_objetivo=magnitud_objetivo, momentum_veto=momentum_veto,
            shadow_mode=self._shadow_mode, risk_verdict=risk_verdict, temporal_verdict=temporal_verdict,
            telemetry_hash=telemetry_hash, manifold_audit=comp.manifold_audit,
            variable_destino=comp.variable_destino,
        )
        if comp.geometric_manifold_shadow:
            master_trace["geometric_manifold_shadow"] = comp.geometric_manifold_shadow

        return build_orchestrated_execution_plan(
            plan_base=plan_base, certeza=certeza, magnitud_objetivo=magnitud_objetivo,
            momentum_veto=momentum_veto, master_trace=master_trace, shadow_mode=self._shadow_mode,
            gamma_exec=self.gamma_exec,
        )
