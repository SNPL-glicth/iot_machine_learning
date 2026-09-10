"""Data models for LiveBotRunner state and execution tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
    last_expert_votes: List[Dict[str, Any]] = field(default_factory=list)
    active_orders: Dict[str, Dict] = field(default_factory=dict)
    current_position: float = 0.0  # Positivo=long, negativo=short
    total_pnl: float = 0.0
    trades_count: int = 0
    last_error: Optional[str] = None


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
    decision_trace: Dict[str, Any]
    telemetry_hash: str
