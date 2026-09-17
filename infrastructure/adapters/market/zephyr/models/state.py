"""Data models for LiveBotRunner state and execution tracking."""

from __future__ import annotations

import time
from dataclasses import InitVar, dataclass, field
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
    positions: dict[str, float] = field(default_factory=dict)
    current_position: InitVar[float | None] = None
    equity: float = 10000.0
    account_blocked: bool = False
    total_pnl: float = 0.0
    trades_count: int = 0
    last_error: str | None = None
    market_closing_soon: bool = False
    market_closed: bool = False
    portfolio_circuit_breaker_tripped: bool = False
    daily_peak_equity: float = 0.0
    is_closing: dict[str, bool] = field(default_factory=dict)
    close_confirmed_at: dict[str, float] = field(default_factory=dict)
    consecutive_losses: dict[str, int] = field(default_factory=dict)
    streak_cooling_until: dict[str, float] = field(default_factory=dict)

    def __post_init__(self, current_position: float | None = None) -> None:
        if isinstance(current_position, (int, float)) and current_position != 0.0:
            if not self.positions:
                self.positions["SPY"] = float(current_position)

    @property
    def current_position(self) -> float:
        """
        Propiedad calculada para compatibilidad regresiva con componentes legacy/tests.
        La única fuente de verdad es self.positions por símbolo.
        """
        if "SPY" in self.positions:
            return self.positions["SPY"]
        if len(self.positions) == 1:
            return next(iter(self.positions.values()))
        return 0.0

    @current_position.setter
    def current_position(self, qty: float) -> None:
        """Setter de compatibilidad: escribe en self.positions para el símbolo principal."""
        if len(self.positions) == 1 and "SPY" not in self.positions:
            sym = next(iter(self.positions.keys()))
            self.positions[sym] = float(qty)
        else:
            self.positions["SPY"] = float(qty)

    @property
    def active_positions_count(self) -> int:
        return sum(1 for qty in self.positions.values() if qty != 0.0)

    def get_position(self, symbol: str) -> float:
        return self.positions.get(symbol.upper(), 0.0)

    def set_position(self, symbol: str, qty: float) -> None:
        self.positions[symbol.upper()] = float(qty)

    def mark_closing(self, symbol: str) -> None:
        """Marca el símbolo como en proceso de cierre antes de enviar órdenes de salida."""
        self.is_closing[symbol.upper()] = True

    def mark_close_confirmed(self, symbol: str) -> None:
        """Registra confirmación real del broker. Mantiene is_closing=True durante el cooldown."""
        self.close_confirmed_at[symbol.upper()] = time.time()

    def is_symbol_closing(self, symbol: str, cooldown_sec: float = 45.0) -> bool:
        """
        Verifica si un símbolo está cerrando o en cooldown obligatorio post-cierre.
        Si el cierre falló (nunca confirmado), is_closing queda permanentemente en True.
        """
        sym = symbol.upper()
        if not self.is_closing.get(sym, False):
            return False
        if sym in self.close_confirmed_at:
            elapsed = time.time() - self.close_confirmed_at[sym]
            if elapsed >= cooldown_sec:
                self.is_closing[sym] = False
                del self.close_confirmed_at[sym]
                return False
            return True
        # En vuelo o falló sin confirmación: mantener bloqueado
        return True

    def record_trade_outcome(
        self,
        symbol: str,
        realized_pnl: float,
        max_consecutive_losses: int = 2,
        cooldown_sec: float = 900.0,
    ) -> bool:
        """
        Registra el resultado de un trade cerrado.
        Si realized_pnl < 0, incrementa racha de pérdidas. Si >= 0, resetea la racha.
        Al acumular max_consecutive_losses, activa una pausa de enfriamiento de cooldown_sec (15 min).
        Retorna True si se activó el cooldown por racha de pérdidas.
        """
        sym = symbol.upper()
        if hasattr(self, "on_trade_outcome") and callable(self.on_trade_outcome):
            try: self.on_trade_outcome(realized_pnl)
            except Exception: pass
        if realized_pnl <= -0.01:
            count = self.consecutive_losses.get(sym, 0) + 1
            self.consecutive_losses[sym] = count
            import logging
            _logger = logging.getLogger(__name__)
            _logger.warning(
                "Consecutive loss recorded for %s: count=%d, realized_pnl=-$%.2f",
                sym, count, abs(realized_pnl),
            )
            if count >= max_consecutive_losses:
                cooling_until = time.time() + cooldown_sec
                self.streak_cooling_until[sym] = cooling_until
                _logger.warning(
                    "CONSECUTIVE LOSS COOLDOWN TRIGGERED [%s]: %d consecutive losses. "
                    "Entry orders vetoed for %.1f minutes (until %s).",
                    sym, count, cooldown_sec / 60.0, time.ctime(cooling_until),
                )
                return True
        else:
            if self.consecutive_losses.get(sym, 0) > 0:
                import logging
                logging.getLogger(__name__).info(
                    "Winning trade on %s (realized_pnl=+$%.2f). Resetting consecutive loss counter to 0.",
                    sym, realized_pnl,
                )
            self.consecutive_losses[sym] = 0
        return False


    def is_in_loss_streak_cooldown(self, symbol: str) -> bool:
        """Verifica si el símbolo está en período de enfriamiento por racha de pérdidas."""
        sym = symbol.upper()
        if sym not in self.streak_cooling_until:
            return False
        if time.time() < self.streak_cooling_until[sym]:
            return True
        # Expiró el enfriamiento por racha
        del self.streak_cooling_until[sym]
        return False


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
