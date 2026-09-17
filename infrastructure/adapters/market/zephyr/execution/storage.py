"""Atomic state persistence routines for LiveBotState."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import time

import aiofiles

from iot_machine_learning.infrastructure.adapters.market.zephyr.models import LiveBotState

logger = logging.getLogger(__name__)


async def save_state(state: LiveBotState, state_path: Path | str | None) -> None:
    """Guarda estado en disco de forma atómica usando archivo temporal."""
    if not state_path:
        return
    try:
        payload = {
            "cycle_count": int(state.cycle_count),
            "last_execution_time": float(state.last_execution_time),
            "last_execution_price": float(state.last_execution_price),
            "last_execution_side": str(state.last_execution_side or ""),
            "last_phi_moe": float(state.last_phi_moe),
            "last_lambda_t": float(state.last_lambda_t),
            "last_phi_ritmo": float(state.last_phi_ritmo),
            "current_position": float(state.current_position),
            "positions": {k: float(v) for k, v in state.positions.items()},
            "total_pnl": float(state.total_pnl),
            "trades_count": int(state.trades_count),
            "consecutive_losses": {k: int(v) for k, v in state.consecutive_losses.items()},
            "streak_cooling_until": {k: float(v) for k, v in state.streak_cooling_until.items()},
            "portfolio_circuit_breaker_tripped": bool(state.portfolio_circuit_breaker_tripped),
            "last_error": str(state.last_error) if state.last_error else None,
            "timestamp": time.time(),
        }
        p = Path(state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = p.with_suffix(".tmp")
        async with aiofiles.open(tmp_p, "w") as f:
            await f.write(json.dumps(payload, indent=2))
        tmp_p.replace(p)
    except Exception as e:
        logger.warning("Failed to save state: %s", e)


async def load_state(state: LiveBotState, state_path: Path | str | None) -> None:
    """Carga estado guardado en disco."""
    if not state_path or not Path(state_path).exists():
        return
    try:
        async with aiofiles.open(state_path, "r") as f:
            data = json.loads(await f.read())
            state.cycle_count = int(data.get("cycle_count", 0))
            for field in ("last_execution_time", "last_execution_price", "last_phi_moe", "last_lambda_t", "last_phi_ritmo", "current_position", "total_pnl"):
                setattr(state, field, data.get(field, 0.0))
            state.positions = data.get("positions", {})
            state.consecutive_losses = data.get("consecutive_losses", {})
            state.streak_cooling_until = data.get("streak_cooling_until", {})
            state.portfolio_circuit_breaker_tripped = bool(data.get("portfolio_circuit_breaker_tripped", False))
            state.last_execution_side = data.get("last_execution_side", "")
            state.trades_count = int(data.get("trades_count", 0))
    except Exception as e:
        logger.warning("Failed to load state: %s", e)
