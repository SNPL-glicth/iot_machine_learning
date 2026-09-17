"""Order sizing, envelope calculation, and submission payload builder for Rosa Roja."""

from __future__ import annotations

import logging
import time
from typing import Any

from infrastructure.ml.engines.rosa_roja.algorithms.domain.execution import ExecutionPlan

logger = logging.getLogger(__name__)


def calculate_unrealized_pnl(
    side: str,
    qty: float,
    avg_entry_price: float,
    current_price: float,
) -> float:
    """Calcula el PnL no realizado con consistencia para posiciones Long y Short."""
    try:
        q, entry, curr = float(qty), float(avg_entry_price), float(current_price)
    except (ValueError, TypeError):
        return 0.0
    if entry <= 0.0 or curr <= 0.0 or q == 0.0:
        return 0.0
    s = str(side).strip().lower() if side is not None else ""
    is_short = s in ("short", "sell") or q < 0.0
    abs_qty = abs(q)
    return (entry - curr) * abs_qty if is_short else (curr - entry) * abs_qty


def build_trajectory_order_payload(
    plan: ExecutionPlan,
    target_sym: str,
    current_price: float,
    equity: float,
    max_position_pct: float,
    lot_size: float,
    min_qty: float,
    max_qty: float,
) -> tuple[dict[str, Any] | None, str]:
    """
    Construye los argumentos para broker.submit_order diferenciando Modo Carrito y Modo Estándar.
    
    Retorna (submit_kwargs, reason_if_skipped).
    """
    envelope = plan.envelope
    if envelope is None or plan.chosen_trajectory is None:
        return None, "Missing_Envelope_Or_Trajectory"

    terminal_state = getattr(plan.chosen_trajectory, "terminal_state", None)
    terminal_price = float(terminal_state.state_vector[0]) if terminal_state is not None and hasattr(terminal_state, "state_vector") and len(terminal_state.state_vector) > 0 else 0.0
    p_side = getattr(plan, "side", None)
    t_side = getattr(plan.chosen_trajectory, "side", None) if getattr(plan, "chosen_trajectory", None) else None
    if isinstance(p_side, str) and p_side:
        side = p_side.lower()
    elif isinstance(t_side, str) and t_side:
        side = t_side.lower()
    else:
        side = "buy" if terminal_price >= 0 else "sell"

    is_micro_mode = equity < 2000.0

    if is_micro_mode and side == "sell":
        logger.info("Micro-mode (< $2000 USD): Skipping short entry for %s (equity=$%.2f)", target_sym, equity)
        return None, "Micro_Mode_Short_Veto"

    step_idx = terminal_state.step_index if terminal_state else 0
    submit_kwargs: dict[str, Any] = {
        "symbol": target_sym,
        "side": side,
        "order_type": "market",
        "time_in_force": "day",
        "client_order_id": f"RR_{step_idx}_{int(time.time()*1000)}_entry",
    }

    if is_micro_mode:
        # Permite asignar hasta max_position_pct del saldo disponible (ej. 85-90% de $20 = $17-$18 USD)
        max_notional_cap = max(1.0, equity - 0.50)
        notional = max(1.0, min(equity * max_position_pct, max_notional_cap))
        submit_kwargs.update({"qty": 0.0, "notional": round(notional, 2), "order_class": None, "take_profit": None, "stop_loss": None})
    else:
        notional = equity * min(envelope.magnitude, max_position_pct)
        qty = min(max_qty, max(min_qty, round(notional / current_price / lot_size) * lot_size))
        stop_pct = float(envelope.bounds.get("stop_pct", 0.0))
        target_pct = float(envelope.bounds.get("target_pct", 0.0))
        eff_tp = target_pct if target_pct > 0 else float(envelope.bounds.get("expected_return", 0.0035))
        eff_sl = stop_pct if stop_pct > 0 else float(envelope.bounds.get("max_loss_pct", 0.0035))

        tp_price = round(current_price * ((1.0 + eff_tp) if side == "buy" else (1.0 - eff_tp)), 2)
        sl_price = round(current_price * ((1.0 - eff_sl) if side == "buy" else (1.0 + eff_sl)), 2)
        take_profit = {"limit_price": str(tp_price)}
        stop_loss = {"stop_price": str(sl_price)}
        submit_kwargs.update({"qty": qty, "order_class": "bracket", "take_profit": take_profit, "stop_loss": stop_loss})

    logger.info(
        "Dispatching Rosa Roja execution plan",
        extra={
            "symbol": target_sym, "action": plan.action, "confidence": plan.global_confidence,
            "side": side, "is_micro_mode": is_micro_mode, "current_price": current_price,
            "notional": submit_kwargs.get("notional"),
        },
    )
    return submit_kwargs, ""
