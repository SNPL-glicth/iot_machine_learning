"""Pipeline runner: ejecución secuencial del núcleo de Rosa Roja y mapeo de resultados."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING
import numpy as np

from .algorithms.domain.execution import ExecutionPlan
from .algorithms.engine import RosaRojaEngine
from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult

if TYPE_CHECKING:
    from iot_machine_learning.infrastructure.ml.moe.experts.rosa_roja_expert import (
        RosaRojaResult,
    )


def sanitize_inputs(
    values: List[float], timestamps: Optional[List[float]]
) -> tuple[List[float], Optional[List[float]]]:
    """Filtra valores nulos, NaN o infinitos de la serie temporal."""
    ts = timestamps if timestamps is not None else [float(i) for i in range(len(values))]
    clean = [
        (v, t)
        for v, t in zip(values, ts)
        if v is not None and not np.isnan(v) and not np.isinf(v)
    ]
    if not clean:
        return [], None
    c_val, c_ts = zip(*clean)
    return list(c_val), list(c_ts) if timestamps is not None else None


def execute_time_series(
    engine: RosaRojaEngine,
    values: List[float],
    timestamps: Optional[List[float]] = None,
) -> tuple[ExecutionPlan, float, Optional[int]]:
    """Alimenta la serie al motor Rosa Roja y obtiene el plan de ejecución final."""
    plan = ExecutionPlan.HOLD(reason="Insufficient_Data")
    invalidation_step: Optional[int] = None
    delta_next = 0.0

    for i in range(1, len(values)):
        dv = values[i] - values[i - 1]
        dt = (timestamps[i] - timestamps[i - 1]) if timestamps else 1.0
        dt = max(dt, 1e-4)
        delta_state = np.array([dv], dtype=np.float64)
        plan = engine.process_event(delta_state=delta_state, delta_time=dt)

    if plan.action == "EXECUTE" and plan.chosen_trajectory and len(plan.chosen_trajectory.movements) > 1:
        delta_next = float(plan.chosen_trajectory.movements[1].delta_state[0])
        invalidation_step = plan.chosen_trajectory.invalidation_step

    return plan, delta_next, invalidation_step


def map_to_prediction_result(
    plan: ExecutionPlan,
    delta_next: float,
    last_value: float,
    invalidation_step: Optional[int],
) -> PredictionResult:
    """Mapea el plan de ejecución a PredictionResult para la interfaz PredictionEngine."""
    reason = plan.veto_details.get("reason", "") if plan.veto_details else ""
    if plan.action == "EXECUTE":
        pred_val = last_value + delta_next
        confidence = float(np.clip(plan.global_confidence, 0.0, 1.0))
        trend = "up" if delta_next > 1e-5 else ("down" if delta_next < -1e-5 else "stable")
    else:
        pred_val = last_value
        confidence = float(np.clip(plan.global_confidence, 0.0, 0.4))
        trend = "stable"

    meta = {
        "engine": "rosa_roja_moe_head",
        "action": plan.action,
        "reason": reason,
        "invalidation_step": invalidation_step,
        "coherence_confidence": plan.global_confidence,
    }
    return PredictionResult(
        predicted_value=pred_val,
        confidence=confidence,
        trend=trend,
        metadata=meta,
    )


def map_to_rosa_roja_result(
    plan: ExecutionPlan,
    delta_next: float,
    invalidation_step: Optional[int],
    s_t: Dict[str, Any],
) -> RosaRojaResult:
    """Mapea el plan a RosaRojaResult para compatibilidad con RosaRojaExpert."""
    from iot_machine_learning.infrastructure.ml.moe.experts.rosa_roja_expert import (
        RosaRojaResult,
    )

    reason = plan.veto_details.get("reason", "") if plan.veto_details else ""
    direction = "up" if delta_next > 1e-5 else ("down" if delta_next < -1e-5 else "stable")
    trajectory_names = (
        [f"step_{m.timestamp:.1f}" for m in plan.chosen_trajectory.movements]
        if plan.chosen_trajectory
        else ["hold"]
    )
    rhythm_score = (
        plan.chosen_trajectory.coherence_score
        if plan.chosen_trajectory
        else 0.5
    )
    return RosaRojaResult(
        trajectory=trajectory_names,
        trajectory_score=plan.global_confidence,
        rhythm_score=rhythm_score,
        lambda_val=0.3,
        theta_entropy=0.5,
        regime_alert=s_t.get("current_regime", "stable"),
        invalidation_step=invalidation_step,
        expected_direction=direction,
        expected_magnitude=abs(delta_next),
        confidence=float(np.clip(plan.global_confidence, 0.0, 1.0)),
        evidence={"action": plan.action, "reason": reason},
        status="ok" if plan.action == "EXECUTE" else "hold",
    )
