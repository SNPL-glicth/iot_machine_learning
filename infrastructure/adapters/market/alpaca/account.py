"""AlpacaAccount -- Gestión de cuenta y posiciones para Alpaca Paper Trading."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional

from iot_machine_learning.infrastructure.adapters.market.alpaca.account_models import (
    AccountSnapshot, Position,
)
from iot_machine_learning.infrastructure.adapters.market.alpaca.account_sync import (
    fetch_account_and_positions,
)
from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client import AlpacaOrderClient

logger = logging.getLogger(__name__)

__all__ = ["Position", "AccountSnapshot", "AlpacaAccount", "create_account"]


class AlpacaAccount:
    """Gestión de cuenta para Alpaca Paper Trading."""

    def __init__(self, client: AlpacaOrderClient, auto_sync_interval: float = 5.0) -> None:
        self._client = client
        self._auto_sync_interval = auto_sync_interval
        self._sync_task: Optional[asyncio.Task] = None
        self._equity = self._cash = self._buying_power = self._portfolio_value = 0.0
        self._pattern_day_trader = self._trading_blocked = False
        self._transfers_blocked = self._account_blocked = False
        self._positions: Dict[str, Position] = {}
        self._last_sync: float = 0.0
        self._sync_running: bool = False
        self._lock = asyncio.Lock()
        logger.info("AlpacaAccount initialized")

    async def start_auto_sync(self) -> None:
        """Inicia sincronización automática periódica."""
        if not self._sync_running:
            self._sync_running = True
            self._sync_task = asyncio.create_task(self._auto_sync_loop())
            logger.info("Auto-sync started")

    async def stop_auto_sync(self) -> None:
        """Detiene sincronización automática."""
        self._sync_running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None
        logger.info("Auto-sync stopped")

    async def sync_now(self) -> None:
        """Fuerza sincronización inmediata."""
        async with self._lock:
            await self._sync()

    async def _ensure_synced(self) -> None:
        if time.time() - self._last_sync > 1.0:
            await self._sync()

    async def get_equity(self) -> float:
        async with self._lock:
            await self._ensure_synced()
            return self._equity

    async def get_cash(self) -> float:
        async with self._lock:
            await self._ensure_synced()
            return self._cash

    async def get_buying_power(self) -> float:
        async with self._lock:
            await self._ensure_synced()
            return self._buying_power

    async def get_portfolio_value(self) -> float:
        async with self._lock:
            await self._ensure_synced()
            return self._portfolio_value

    async def get_position(self, symbol: str) -> float:
        """Posición neta (positivo=long, negativo=short)."""
        async with self._lock:
            await self._ensure_synced()
            pos = self._positions.get(symbol.upper())
            return (pos.qty if pos.side == "long" else -pos.qty) if pos else 0.0

    async def get_position_details(self, symbol: str) -> Optional[Position]:
        """Detalles completos de posición."""
        async with self._lock:
            await self._ensure_synced()
            return self._positions.get(symbol.upper())

    async def get_all_positions(self) -> Dict[str, Position]:
        """Todas las posiciones."""
        async with self._lock:
            await self._ensure_synced()
            return self._positions.copy()

    async def get_unrealized_pl(self, symbol: Optional[str] = None) -> float:
        """PnL no realizado de una posición o total."""
        async with self._lock:
            await self._ensure_synced()
            if symbol:
                pos = self._positions.get(symbol.upper())
                return pos.unrealized_pl if pos else 0.0
            return sum(p.unrealized_pl for p in self._positions.values())

    async def get_account_status(self) -> Dict[str, Any]:
        """Estado de la cuenta (bloqueos, PDT, etc.)."""
        async with self._lock:
            await self._ensure_synced()
            return {
                "pattern_day_trader": self._pattern_day_trader,
                "trading_blocked": self._trading_blocked,
                "transfers_blocked": self._transfers_blocked,
                "account_blocked": self._account_blocked,
            }

    async def is_tradeable(self) -> bool:
        """Verifica si la cuenta puede operar."""
        st = await self.get_account_status()
        return not (st["trading_blocked"] or st["account_blocked"] or st["transfers_blocked"])

    async def get_snapshot(self) -> AccountSnapshot:
        """Snapshot completo de la cuenta."""
        async with self._lock:
            await self._ensure_synced()
            return AccountSnapshot(
                equity=self._equity, cash=self._cash, buying_power=self._buying_power,
                portfolio_value=self._portfolio_value, pattern_day_trader=self._pattern_day_trader,
                trading_blocked=self._trading_blocked, transfers_blocked=self._transfers_blocked,
                account_blocked=self._account_blocked, positions=self._positions.copy(),
                timestamp=time.time(),
            )

    async def _auto_sync_loop(self) -> None:
        while self._sync_running:
            try:
                await asyncio.sleep(self._auto_sync_interval)
                if self._sync_running:
                    await self._sync()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Auto-sync error", extra={"error": str(e)})
                await asyncio.sleep(5)

    async def _sync(self) -> None:
        try:
            f, self._positions = await fetch_account_and_positions(self._client)
            self._equity, self._cash = f["equity"], f["cash"]
            self._buying_power, self._portfolio_value = f["buying_power"], f["portfolio_value"]
            self._pattern_day_trader = f["pattern_day_trader"]
            self._trading_blocked, self._transfers_blocked = f["trading_blocked"], f["transfers_blocked"]
            self._account_blocked = f["account_blocked"]
            self._last_sync = time.time()
            logger.debug("Account synced", extra={"equity": self._equity, "positions": len(self._positions)})
        except Exception as e:
            logger.error("Account sync failed", extra={"error": str(e)})
            raise


async def create_account(client: AlpacaOrderClient, auto_sync: bool = True) -> AlpacaAccount:
    """Factory para crear y configurar cuenta."""
    account = AlpacaAccount(client)
    if auto_sync:
        await account.start_auto_sync()
    return account