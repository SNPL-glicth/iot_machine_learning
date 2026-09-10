"""Data models for LiveBotRunner state and execution tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LiveBotState:
    """Estado interno del bot para persistencia y recuperación."""
    cycle_count: int = 0
    last_execution_time: float = 0.0
    last_execution_price: float = 0.0
    last_execution_side: str = ""
    last_phi_moe: float = 0.0
    last_lambda_t: float = 0.0
    last_phi_ritmo: float = 0.0
    last_expert_votes: list[dict[str, Any]] = field(default_factory=list)
    active_orders: dict[str, dict] = field(default_factory=dict)
    current_position: float = 0.0  # Positivo=long, negativo=short
    positions: dict[str, float] = field(default_factory=dict)
    total_pnl: float = 0.0
    trades_count: int = 0
    last_error: str | None = None
    market_closing_soon: bool = False
    market_closed: bool = False
    portfolio_circuit_breaker_tripped: bool = False
    daily_peak_equity: float = 0.0

    @property
    def active_positions_count(self) -> int:
        return sum(1 for qty in self.positions.values() if qty != 0.0)

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol.upper(), self.current_position if symbol.upper() == "SPY" else 0.0)

    def set_position(self, symbol: str, qty: float) -> None:
        self.positions[symbol.upper()] = qty
        self.current_position = qty


@dataclass
class ExecutionContext:
    """Contexto de una ejecución para auditoría (ISO 22989)."""
    timestamp: float
    phi_moe: float
    lambda_t: float
    phi_ritmo: float
    action: str
    side: str
    qty: float
    price: float
    order_type: str
    decision_trace: dict[str, Any]
    telemetry_hash: str
