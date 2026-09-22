"""Native Market Execution Handler implementing Rosa Roja's ExecutionPort."""

from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from domain.entities.rosa_roja.execution import ExecutionPlan
from domain.ports.rosa_roja.execution_port import ExecutionPort
from iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja_execution.liquidation import (
    execute_emergency_flush, execute_trailing_and_stop_loss_check, is_close_confirmed,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja_execution.order_builder import (
    build_trajectory_order_payload, calculate_unrealized_pnl,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.risk.trailing_profit_manager import (
    TrailingProfitConfig, TrailingProfitManager,
)

logger = logging.getLogger(__name__)


class PriceUnavailableError(RuntimeError):
    """Raised when no valid market reference price is available for an asset."""


class BrokerClientProtocol(Protocol):
    """Minimal broker interface for order dispatch."""
    async def submit_order(self, symbol: str, side: str, order_type: str, qty: float, price: float | None = None,
                           stop_price: float | None = None, time_in_force: str = "GTC", client_order_id: str | None = None) -> dict[str, Any]: ...
    async def cancel_order(self, order_id: str) -> bool: ...
    async def cancel_all_orders(self, symbol: str | None = None) -> int: ...
    async def get_position(self, symbol: str) -> float: ...
    async def close_position(self, symbol: str) -> bool: ...


class RosaRojaMarketExecutionHandler(ExecutionPort):
    """Native Market execution handler fulfilling Rosa Roja's ExecutionPort contract."""

    def __init__(
        self, broker_client: Any, account_equity: float, symbol: str,
        lot_size: float = 1.0, min_qty: float = 0.01, max_qty: float = float("inf"), max_position_pct: float = 1.0,
        trailing_config: TrailingProfitConfig | None = None, max_stop_loss_usd: float = 10.0,
        state: Any | None = None, max_consecutive_losses: int = 2, consecutive_loss_cooldown_sec: float = 900.0,
    ):
        self._broker, self._equity, self._symbol = broker_client, account_equity, symbol
        self._lot_size, self._min_qty, self._max_qty, self._max_position_pct = lot_size, min_qty, max_qty, max_position_pct
        self._max_stop_loss_usd, self._state = max_stop_loss_usd, state
        self._max_consecutive_losses, self._consecutive_loss_cooldown_sec = max_consecutive_losses, consecutive_loss_cooldown_sec
        self._active_orders: dict[str, dict[str, Any]] = {}
        self._trailing_config = trailing_config
        self._trailing_manager = TrailingProfitManager(trailing_config)
        self._trailing_managers: dict[str, TrailingProfitManager] = {symbol.upper(): self._trailing_manager}
        self._cached_positions: dict[str, dict[str, Any] | None] = {}
        self._last_pos_checks: dict[str, float] = {}
        self._last_known_prices: dict[str, tuple[float, float]] = {}

    def _mark_closing(self, symbol: str) -> None:
        if self._state is not None and hasattr(self._state, "mark_closing"): self._state.mark_closing(symbol)

    def _mark_close_confirmed(self, symbol: str) -> None:
        if self._state is not None and hasattr(self._state, "mark_close_confirmed"): self._state.mark_close_confirmed(symbol)

    def _record_trade_outcome(self, symbol: str, realized_pnl: float) -> None:
        if self._state is not None and hasattr(self._state, "record_trade_outcome"):
            self._state.record_trade_outcome(symbol, realized_pnl=realized_pnl, max_consecutive_losses=self._max_consecutive_losses, cooldown_sec=self._consecutive_loss_cooldown_sec)

    def _get_trailing_manager(self, sym: str) -> TrailingProfitManager:
        s = sym.upper()
        if s not in self._trailing_managers:
            if self._equity < 2000.0 and self._trailing_config:
                scale = max(0.001, self._equity / 10000.0)
                cfg = TrailingProfitConfig(
                    activation_pnl_usd=max(0.02, round(self._trailing_config.activation_pnl_usd * scale, 3)),
                    min_giveback_usd=max(0.01, round(self._trailing_config.min_giveback_usd * scale, 3)),
                    max_giveback_usd=max(0.01, round(self._trailing_config.max_giveback_usd * scale, 3)),
                    giveback_ratio=self._trailing_config.giveback_ratio,
                )
                self._trailing_managers[s] = TrailingProfitManager(cfg)
            else:
                self._trailing_managers[s] = TrailingProfitManager(self._trailing_config)
        return self._trailing_managers[s]

    async def dispatch_execution(self, plan: ExecutionPlan, symbol: str | None = None) -> bool:
        if plan.action == "HOLD": return True
        target_sym = (symbol or getattr(plan, "symbol", None) or self._symbol).upper()
        if plan.action in ("EMERGENCY_FLUSH", "CLOSE") or plan.regime_alert:
            reason = plan.veto_details.get("reason", "RegimeAlert_Triggered") if getattr(plan, "veto_details", None) else "RegimeAlert_Triggered"
            logger.warning("Executing ZENIN directive %s for %s: %s", plan.action, target_sym, reason)
            await self.trigger_emergency_flush(reason=reason, symbol=target_sym)
            return False
        if plan.action == "EXECUTE":
            return await self._execute_trajectory_orders(plan, symbol=target_sym)
        return False

    async def _execute_trajectory_orders(self, plan: ExecutionPlan, symbol: str | None = None) -> bool:
        try:
            target_sym = (symbol or getattr(plan, "symbol", None) or self._symbol).upper()
            current_price = self._get_reference_price(target_sym)
            submit_kwargs, reason = build_trajectory_order_payload(
                plan, target_sym, current_price, self._equity, self._max_position_pct, self._lot_size, self._min_qty, self._max_qty,
            )
            if not submit_kwargs:
                if reason: logger.debug("Skipping dispatch for %s: %s", target_sym, reason)
                return False

            entry_order = await self._broker.submit_order(**submit_kwargs)
            self._track_order(entry_order, symbol=target_sym)
            self._cached_positions[target_sym] = None
            return True
        except PriceUnavailableError as pe:
            logger.error("Aborting order dispatch for %s: %s", symbol or self._symbol, pe)
            return False
        except Exception as e:
            logger.error("Failed to dispatch execution plan: %s", e, exc_info=True)
            return False

    def _get_reference_price(self, symbol: str, max_stale_age_sec: float = 60.0) -> float:
        sym = symbol.upper()
        if hasattr(self, "get_reference_price_callback") and callable(self.get_reference_price_callback):
            cb_price = self.get_reference_price_callback(sym)
            if cb_price is not None and cb_price > 0:
                self._last_known_prices[sym] = (float(cb_price), time.time())
                return float(cb_price)
        cached = self._last_known_prices.get(sym)
        if cached and (time.time() - cached[1]) <= max_stale_age_sec and cached[0] > 0: return cached[0]
        raise PriceUnavailableError(f"Market reference price unavailable for symbol '{symbol}'")

    def _track_order(self, order_response: Any, symbol: str | None = None) -> None:
        order_id = getattr(order_response, "id", None) or getattr(order_response, "client_order_id", None) or (order_response.get("id") if isinstance(order_response, dict) else None)
        if order_id:
            sym = (symbol or self._symbol).upper()
            self._active_orders.setdefault(sym, {})[str(order_id)] = order_response

    def _is_close_confirmed(self, close_resp: Any) -> bool: return is_close_confirmed(close_resp)

    async def trigger_emergency_flush(self, reason: str, symbol: str | None = None) -> None:
        sym = (symbol or self._symbol).upper()
        await execute_emergency_flush(self._broker, sym, reason, self._active_orders, self._mark_closing, self._mark_close_confirmed, self._state)
        self._cached_positions[sym] = None

    def update_equity(self, new_equity: float) -> None:
        old_eq = self._equity
        self._equity = new_equity
        if (old_eq < 2000.0 <= new_equity) or (old_eq >= 2000.0 > new_equity): self._trailing_managers.clear()

    def get_active_orders(self, symbol: str | None = None) -> dict[str, Any]:
        if symbol is not None: return self._active_orders.get(symbol.upper(), {}).copy()
        flattened = {}
        for s_orders in self._active_orders.values(): flattened.update(s_orders)
        return flattened

    async def check_trailing_profit(self, current_price: float, symbol: str | None = None) -> bool:
        sym = (symbol or self._symbol).upper()
        if not self._broker or current_price <= 0: return False
        try:
            return await execute_trailing_and_stop_loss_check(
                self._broker, sym, current_price, self._equity, self._max_stop_loss_usd, self._get_trailing_manager(sym),
                self._cached_positions, self._last_pos_checks, self._mark_closing, self._mark_close_confirmed, self._record_trade_outcome, self._active_orders,
            )
        except Exception as e:
            logger.debug("Error checking trailing profit for %s: %s", sym, e)
            return False
