"""Execution plan builder for Master Equation Orchestrator."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from infrastructure.ml.engines.rosa_roja.algorithms.domain.execution import (
    ActionEnvelope,
    ExecutionPlan,
)
from infrastructure.ml.master_engine.master_equation import compute_magnitud_objetivo


def extract_expert_target_magnitude(
    jury: Sequence[Any],
    plan_base: ExecutionPlan,
) -> float:
    """Extracts expert predictions from jury and computes weighted target magnitude."""
    traj = plan_base.chosen_trajectory
    if traj is None:
        return 0.0

    fallback_mag = abs(float(traj.terminal_state.state_vector[0]))
    predictions: list[float] = []
    weights: list[float] = []

    for expert in jury:
        w = float(getattr(expert, "weight", 1.0))
        engine = getattr(expert, "_engine", None)
        if engine is not None and hasattr(engine, "predict") and traj.movements:
            try:
                vals = [float(m.delta_state[0]) for m in traj.movements]
                res = engine.predict(vals)
                predictions.append(abs(float(res.predicted_value)))
                weights.append(w)
            except Exception:
                pass

    return compute_magnitud_objetivo(
        predicciones=predictions if predictions else [fallback_mag],
        pesos=weights if weights else [1.0],
        default=fallback_mag,
    )


def build_orchestrated_execution_plan(
    plan_base: ExecutionPlan,
    certeza: float,
    magnitud_objetivo: float,
    momentum_veto: float,
    master_trace: dict[str, Any],
    shadow_mode: bool,
    gamma_exec: float = 0.5,
) -> ExecutionPlan:
    """Build final ExecutionPlan adhering to active or shadow mode execution directives."""
    # 1. Shadow Mode: Pure observation without altering execution action or base confidence
    if shadow_mode:
        envelope = plan_base.envelope
        if envelope is not None:
            envelope = ActionEnvelope(
                magnitude=envelope.magnitude,
                bounds=envelope.bounds,
                max_steps=envelope.max_steps,
                metadata={**envelope.metadata, "decision_trace": master_trace},
                decision_trace=master_trace,
            )

        if plan_base.action == "EXECUTE":
            return ExecutionPlan.EXECUTE(
                trajectory=plan_base.chosen_trajectory,
                confidence=plan_base.global_confidence,
                envelope=envelope,
                invalidation_step=plan_base.invalidation_step,
            )

        if plan_base.action == "EMERGENCY_FLUSH":
            reason = (
                plan_base.veto_details.get("reason", "Emergency Flush")
                if isinstance(plan_base.veto_details, dict)
                else "Emergency Flush"
            )
            return ExecutionPlan(
                action="EMERGENCY_FLUSH",
                chosen_trajectory=None,
                global_confidence=0.0,
                envelope=None,
                invalidation_step=None,
                regime_alert=True,
                veto_details={"reason": reason, "decision_trace": master_trace},
            )

        details = dict(plan_base.veto_details) if isinstance(plan_base.veto_details, dict) else {}
        details["decision_trace"] = master_trace
        return ExecutionPlan.HOLD(
            reason=details.get("reason", "Hold_Evaluated"),
            details=details,
        )

    # 2. Active Mode (shadow_mode=False): Master Equation takes full control
    if plan_base.action == "EMERGENCY_FLUSH":
        return ExecutionPlan(
            action="EMERGENCY_FLUSH",
            chosen_trajectory=None,
            global_confidence=0.0,
            envelope=None,
            invalidation_step=None,
            regime_alert=True,
            veto_details={"reason": "Emergency_Flush_Triggered", "decision_trace": master_trace},
        )

    if certeza < gamma_exec:
        return ExecutionPlan.HOLD(
            reason="Certeza_Below_Gamma_Exec",
            details={"reason": "Certeza_Below_Gamma_Exec", "decision_trace": master_trace},
        )

    if momentum_veto <= 0.0:
        return ExecutionPlan.HOLD(
            reason="Momentum_Veto_Triggered",
            details={"reason": "Momentum_Veto_Triggered", "decision_trace": master_trace},
        )

    if plan_base.chosen_trajectory is None:
        return ExecutionPlan.HOLD(
            reason="No_Trajectory_For_Execution",
            details={"reason": "No_Trajectory_For_Execution", "decision_trace": master_trace},
        )

    base_bounds = plan_base.envelope.bounds if plan_base.envelope else {}
    base_max_steps = plan_base.envelope.max_steps if plan_base.envelope else 15
    active_envelope = ActionEnvelope(
        magnitude=float(magnitud_objetivo),
        bounds=base_bounds,
        max_steps=base_max_steps,
        metadata={"decision_trace": master_trace},
        decision_trace=master_trace,
    )

    return ExecutionPlan.EXECUTE(
        trajectory=plan_base.chosen_trajectory,
        confidence=float(certeza),
        envelope=active_envelope,
        invalidation_step=plan_base.invalidation_step,
    )
