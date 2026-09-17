"""Observation pipeline and execution routing for LiveBotRunner obeying ZENIN Master Equation.

Enforces absolute mathematical supremacy of MasterEquationOrchestrator through the
dedicated MasterEngineAdapter, PlanTranslator, and MasterTelemetryPacker.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from iot_machine_learning.infrastructure.adapters.market.zephyr.master_engine_adapter import (
    MasterEngineAdapter,
    MasterTelemetryPacker,
    OrderDirective,
    PlanTranslator,
    extract_decision_trace_metrics,
    resolve_plan_action_trigger,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.models import LiveBotState

logger = logging.getLogger(__name__)

# Re-export for backward compatibility with existing tests
__all__ = [
    "resolve_plan_action_trigger",
    "extract_decision_trace_metrics",
    "process_observation_pipeline",
]


def _get_or_create_adapter(runner: Any, sym: str, engine: Any) -> MasterEngineAdapter:
    """Retrieves or creates a cached MasterEngineAdapter for the given symbol."""
    raw = getattr(runner, "_engine_adapters", None)
    if not isinstance(raw, dict):
        adapter = MasterEngineAdapter(engine, symbol=sym, config=getattr(runner, "config", None))
        try:
            runner._engine_adapters = {sym: adapter}
        except Exception:
            pass
        return adapter
    if sym not in raw or raw[sym].engine is not engine:
        raw[sym] = MasterEngineAdapter(engine, symbol=sym, config=getattr(runner, "config", None))
    return raw[sym]


async def process_observation_pipeline(obs: Any, runner: Any) -> None:
    """Executes feature extraction, ZENIN Master Equation inference, and sovereign order dispatch.

    Args:
        obs: Market observation (Quote, Trade, Candle, OrderBookSnapshot).
        runner: The active LiveBotRunner hosting feeds, state, and execution handlers.
    """
    sym = getattr(obs, "symbol", runner.config.symbol)
    extractor = runner._symbol_extractors.get(sym, runner._feature_extractor)
    engine = runner._symbol_engines.get(sym, runner._engine)
    if engine is None:
        return

    delta_state, delta_time = extractor.process(obs)
    mid = runner._get_current_mid(sym)

    # 1. Sovereign Evaluation via MasterEngineAdapter (Zero Silent Errors)
    adapter = _get_or_create_adapter(runner, sym, engine)
    current_pos = runner._state.get_position(sym) if hasattr(runner._state, "get_position") else 0.0
    plan = adapter.evaluate(delta_state, delta_time, current_position=current_pos)

    # 2. Canonical Translation into Order Directive
    directive: OrderDirective = PlanTranslator.translate(plan)
    runner._state.last_phi_moe = directive.confidence
    runner._state.last_phi_ritmo = (
        plan.chosen_trajectory.coherence_score if getattr(plan, "chosen_trajectory", None) else 0.0
    )

    # 3. Telemetry and Diagnostic Extraction
    shadow_str, scores = MasterTelemetryPacker.pack(plan, runner._state)
    now = time.time()
    last_eval = runner._last_eval_logs.get(sym, 0.0)
    if now - last_eval >= 5.0 or directive.trigger_code != 0:
        runner._last_eval_logs[sym] = now
        logger.info(
            "Evaluation [%s] trigger=%s(%d) reason='%s' mid=%.2f phi_moe=%.3f trades=%d %s",
            sym,
            directive.side.upper(),
            directive.trigger_code,
            directive.veto_reason,
            mid,
            runner._state.last_phi_moe,
            runner._state.trades_count,
            shadow_str,
        )

    # 4. Route Directive (Emergency Flush, Hold, Execute)
    if directive.is_flush:
        if runner._handler:
            await runner._handler.dispatch_execution(plan, symbol=sym)
            runner._state.set_position(sym, 0.0)
        return

    if directive.is_hold:
        adapter.cancel_active_trajectory()
        return

    # 5. Execute Order under Master Equation Dictate
    exec_plan = PlanTranslator.prepare_executable_plan(plan, directive.side)
    if not runner._can_execute(exec_plan, symbol=sym):
        adapter.cancel_active_trajectory()
        return

    dispatched = False
    if runner._handler:
        dispatched = await runner._handler.dispatch_execution(exec_plan, symbol=sym)

    if dispatched:
        runner._state.last_execution_time, runner._state.last_execution_price = time.time(), mid
        runner._state.trades_count += 1
        await runner._log_execution(exec_plan)
        if hasattr(runner, "_weaviate_store"):
            runner._weaviate_store.log_execution(
                MasterTelemetryPacker.create_weaviate_execution_record(
                    sym, directive.side, runner.config.lot_size, mid
                )
            )
        await runner._save_state()
    else:
        adapter.cancel_active_trajectory()

    # 6. Periodic Cognitive Telemetry Persistence to Weaviate
    if hasattr(runner, "_weaviate_store") and (
        now - runner._last_eval_logs.get(f"{sym}_weaviate", 0.0) > 10.0 or directive.trigger_code != 0
    ):
        runner._last_eval_logs[f"{sym}_weaviate"] = now
        runner._weaviate_store.log_telemetry(
            MasterTelemetryPacker.create_weaviate_telemetry_record(
                sym=sym,
                mid=mid,
                phi_moe=runner._state.last_phi_moe,
                lambda_t=getattr(runner._state, "last_lambda_t", 0.0),
                trigger_side=directive.side,
                trigger_code=directive.trigger_code,
                reason=directive.veto_reason,
                scores=scores,
            )
        )
