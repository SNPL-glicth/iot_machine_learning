"""Master Equation Orchestrator (ZENIN v2.2 specification).

Integrates the Rosa Roja trajectory engine, Stochastic Risk Engine adapter,
and Fractal Chronometric Engine adapter into a unified top-level orchestrator.
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np

from infrastructure.ml.engines.rosa_roja.algorithms.domain.execution import ExecutionPlan
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
    """Top-level master orchestrator executing the Master Equation:
        Φ_certeza = I(CVaR_t ≤ L_max) · Λ(t) · Φ_epistémica
    """

    def __init__(
        self,
        rosa_roja_engine: RosaRojaEngine,
        risk_adapter: Any,
        temporal_adapter: Any,
        *,
        shadow_mode: bool = True,
        tau_mom: float = 0.5,
        sigma_mom: float = 0.001,
    ):
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
        """Process market event transition ΔS through the Master Equation pipeline."""
        # 1. Base trajectory generation and epistemic jury evaluation
        plan_base = self._rosa_roja.process_event(delta_state, delta_time)
        base_trace = extract_base_trace(plan_base)
        phi_moe_base = float(base_trace.get("phi_moe", plan_base.global_confidence))

        log_return = float(delta_state[0]) if len(delta_state) > 0 else 0.0
        book_imbalance = float(delta_state[5]) if len(delta_state) > 5 else 0.0

        # 2. Stochastic Risk Adapter: tolerance tube and binary CVaR veto
        risk_verdict: dict[str, Any] = {}
        try:
            risk_verdict = self._risk_adapter.record_observation(
                return_signal=log_return, delta_time=delta_time, log_return=log_return
            )
        except Exception as exc:
            logger.debug("risk_adapter_error: %s", exc)

        i_cvar = float(risk_verdict.get("veto_riesgo", 1.0))

        # 3. Fractal Chronometric Adapter: rhythm synchrony and velocity derivatives
        temporal_verdict: dict[str, Any] = {}
        try:
            temporal_verdict = self._temporal_adapter.record_observation(
                current_price=log_return, book_imbalance=book_imbalance
            )
        except Exception as exc:
            logger.debug("temporal_adapter_error: %s", exc)

        lambda_t_crono = float(temporal_verdict.get("lambda_crono", 1.0))
        ds_dt = float(temporal_verdict.get("dS_dt", abs(log_return / max(1e-6, delta_time))))
        dr_dt = float(temporal_verdict.get("dR_dt", 0.01))

        # 4. Pure Mathematical Computations (Pilar 1)
        certeza = compute_certeza(
            i_cvar=i_cvar,
            ds_dt=ds_dt,
            dr_dt=dr_dt,
            certeza_epistemica=phi_moe_base,
            sigma_dr=1.0,
        )

        jury = getattr(self._rosa_roja, "_jury", [])
        magnitud_objetivo = extract_expert_target_magnitude(jury, plan_base)

        ds_dt_signed = float(np.sign(log_return) * ds_dt) if log_return != 0.0 else float(ds_dt)
        momentum_veto = compute_momentum_veto(
            ds_dt_ema=ds_dt_signed,
            magnitud=magnitud_objetivo,
            tau_mom=self._tau_mom,
            sigma_mom=self._sigma_mom,
        )

        # 5. Assemble Trace and Plan (Pilar 2 & 3)
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
