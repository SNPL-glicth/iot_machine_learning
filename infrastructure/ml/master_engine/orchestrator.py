"""Master Equation Orchestrator (Phase-Space Resonance & Wave Superposition).

Integrates the Rosa Roja trajectory engine, continuous stochastic risk adapter,
and fractal chronometric adapter into an agnostic top-level orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from core.parameters.numerical_constants import EPSILON
from domain.entities.rosa_roja.execution import ExecutionPlan
from infrastructure.ml.engines.rosa_roja.algorithms.engine import RosaRojaEngine
from infrastructure.ml.master_engine.master_equation import (
    compute_certeza,
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
    """Top-level master orchestrator executing continuous wave resonance:
        Φ_certeza = a_risk(t) · Λ(t) · Φ_epistémica
    """

    def __init__(
        self,
        rosa_roja_engine: RosaRojaEngine,
        risk_adapter: Any,
        temporal_adapter: Any,
        *,
        shadow_mode: bool = False,
        tau_mom: float = 0.5,
        sigma_mom: float = 0.001,
    ) -> None:
        self._rosa_roja = rosa_roja_engine
        self._risk_adapter = risk_adapter
        self._temporal_adapter = temporal_adapter
        self._shadow_mode = shadow_mode
        self._tau_mom = tau_mom
        self._sigma_mom = sigma_mom

    @property
    def gamma_exec(self) -> float:
        return getattr(self._rosa_roja, "gamma_exec", 0.5)

    @property
    def state_machine(self) -> Any:
        return self._rosa_roja.state_machine

    def cancel_active_trajectory(self) -> None:
        self._rosa_roja.cancel_active_trajectory()

    def reset(self) -> None:
        self._rosa_roja.reset()
        if hasattr(self._risk_adapter, "reset"):
            self._risk_adapter.reset()
        if hasattr(self._temporal_adapter, "reset"):
            self._temporal_adapter.reset()

    def process_event(self, delta_state: np.ndarray, delta_time: float) -> ExecutionPlan:
        """Process agnostic state transition ΔS through the resonance pipeline."""
        # 1. Base trajectory generation and epistemic jury evaluation
        plan_base = self._rosa_roja.process_event(delta_state, delta_time)
        base_trace = extract_base_trace(plan_base)
        phi_moe_base = float(base_trace.get("phi_moe_base", base_trace.get("phi_moe", plan_base.global_confidence)))

        # 2. Agnostic Phase-Space Kinematics (Zero Hardcoded Financial Slicing)
        arr = np.asarray(delta_state, dtype=np.float64).flatten()
        state_norm = float(np.linalg.norm(arr)) if arr.size > 0 else 0.0
        dt = max(EPSILON.DIVISION, float(delta_time))
        velocity = state_norm / dt
        state_dispersion = float(np.std(arr)) if arr.size > 1 else max(state_norm, 0.0001)

        # 3. Stochastic Risk Adapter: continuous tolerance tube & risk amplitude
        risk_verdict: dict[str, Any] = {}
        try:
            risk_verdict = self._risk_adapter.record_observation(
                return_signal=state_norm, delta_time=dt, log_return=state_norm
            )
        except Exception as exc:
            logger.debug("risk_adapter_error: %s", exc)

        i_cvar = float(risk_verdict.get("veto_riesgo", 1.0))

        # 4. Fractal Chronometric Adapter: phase synchrony & rate of change
        temporal_verdict: dict[str, Any] = {}
        try:
            temporal_verdict = self._temporal_adapter.record_observation(
                current_price=state_norm, price_velocity=velocity
            )
        except Exception as exc:
            logger.debug("temporal_adapter_error: %s", exc)

        lambda_t_crono = float(temporal_verdict.get("lambda_crono", 1.0))
        ds_dt = float(temporal_verdict.get("dS_dt", velocity))
        dr_dt = float(temporal_verdict.get("dR_dt", max(state_dispersion, 0.0001)))

        # 5. Continuous Wave Resonance Computations
        effective_sigma_dr = max(0.1, state_dispersion * 2.0)
        certeza = compute_certeza(
            i_cvar=i_cvar,
            ds_dt=ds_dt,
            dr_dt=dr_dt,
            certeza_epistemica=phi_moe_base,
            sigma_dr=effective_sigma_dr,
        )

        jury = getattr(self._rosa_roja, "_jury", [])
        magnitud_objetivo = extract_expert_target_magnitude(jury, plan_base)

        # 6. Agnostic Directional Gradient Projection
        traj = plan_base.chosen_trajectory
        if traj is not None and hasattr(traj, "directions") and len(traj.directions) > 0 and state_norm > 0:
            state_dir = arr / state_norm
            ref_dir = traj.directions[0]
            if ref_dir.shape == state_dir.shape:
                alignment = float(np.clip(np.dot(state_dir, ref_dir), -1.0, 1.0))
            else:
                alignment = 1.0
        else:
            alignment = 1.0

        ds_dt_projected = float(alignment * ds_dt)
        momentum_veto = compute_momentum_veto(
            ds_dt_ema=ds_dt_projected,
            magnitud=magnitud_objetivo,
            tau_mom=self._tau_mom,
            sigma_mom=self._sigma_mom,
            sigma_market=state_dispersion,
        )

        # 7. Assemble Trace and Plan
        telemetry_hash = compute_telemetry_hash(delta_state, delta_time)
        master_trace = assemble_master_trace(
            base_trace=base_trace,
            phi_moe_base=phi_moe_base,
            i_cvar=i_cvar,
            lambda_t_crono=lambda_t_crono,
            certeza=certeza,
            magnitud_objetivo=magnitud_objetivo,
            momentum_veto=momentum_veto,
            shadow_mode=self._shadow_mode,
            risk_verdict=risk_verdict,
            temporal_verdict=temporal_verdict,
            telemetry_hash=telemetry_hash,
        )

        return build_orchestrated_execution_plan(
            plan_base=plan_base,
            certeza=certeza,
            magnitud_objetivo=magnitud_objetivo,
            momentum_veto=momentum_veto,
            master_trace=master_trace,
            shadow_mode=self._shadow_mode,
            gamma_exec=self.gamma_exec,
        )
