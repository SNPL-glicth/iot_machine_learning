"""AlpacaAccount -- Gestión de cuenta y posiciones para Alpaca Paper Trading."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client import AlpacaOrderClient

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Posición abierta."""
    symbol: str
    side: str          # "long" o "short"
    qty: float         # Cantidad (positiva)
    avg_entry_price: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float
    lastday_price: float
    change_today: float
    asset_id: str
    asset_class: str
    exchange: str
    last_update: float = field(default_factory=time.time)


@dataclass
class AccountSnapshot:
    """Snapshot completo de la cuenta."""
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    positions: Dict[str, Position]
    timestamp: float


class AlpacaAccount:
    """
    Gestión de cuenta para Alpaca Paper Trading.

    Responsabilidades:
    - Sincronización de balances y posiciones
    - Cálculo de PnL (realizado + no realizado)
    - Cálculo de equity, buying power, cash
    - Verificación de restricciones de cuenta
    """

    def __init__(
        self,
        client: AlpacaOrderClient,
        auto_sync_interval: float = 5.0,
    ):
        self._client = client
        self._auto_sync_interval = auto_sync_interval
        self._sync_task: Optional[asyncio.Task] = None

        # Estado de cuenta
        self._equity: float = 0.0
        self._cash: float = 0.0
        self._buying_power: float = 0.0
        self._portfolio_value: float = 0.0
        self._pattern_day_trader: bool = False
        self._trading_blocked: bool = False
        self._transfers_blocked: bool = False
        self._account_blocked: bool = False
        self._positions: Dict[str, Position] = {}
        self._last_sync: float = 0.0
        self._sync_running = False
        self._lock = asyncio.Lock()

        logger.info("AlpacaAccount initialized")

    async def start_auto_sync(self) -> None:
        """Inicia sincronización automática periódica."""
        if self._sync_running:
            return
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
        logger.info("Auto-sync stopped")

    async def sync_now(self) -> None:
        """Fuerza sincronización inmediata."""
        async with self._lock:
            await self._sync()

    async def get_equity(self) -> float:
        """Equity total de la cuenta (portfolio value)."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return self._equity

    async def get_cash(self) -> float:
        """Cash disponible."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return self._cash

    async def get_buying_power(self) -> float:
        """Poder de compra disponible."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return self._buying_power

    async def get_portfolio_value(self) -> float:
        """Valor total del portafolio."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return self._portfolio_value

    async def get_position(self, symbol: str) -> float:
        """Posición neta (positivo=long, negativo=short)."""
        sym = symbol.upper()
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            pos = self._positions.get(sym)
            if pos:
                return pos.qty if pos.side == "long" else -pos.qty
            return 0.0

    async def get_position_details(self, symbol: str) -> Optional[Position]:
        """Detalles completos de posición."""
        sym = symbol.upper()
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return self._positions.get(sym)

    async def get_all_positions(self) -> Dict[str, Position]:
        """Todas las posiciones."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return self._positions.copy()

    async def get_unrealized_pl(self, symbol: Optional[str] = None) -> float:
        """PnL no realizado de una posición o total."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            if symbol:
                sym = symbol.upper()
                pos = self._positions.get(sym)
                return pos.unrealized_pl if pos else 0.0
            return sum(p.unrealized_pl for p in self._positions.values())

    async def get_account_status(self) -> Dict[str, Any]:
        """Estado de la cuenta (bloqueos, PDT, etc.)."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return {
                "pattern_day_trader": self._pattern_day_trader,
                "trading_blocked": self._trading_blocked,
                "transfers_blocked": self._transfers_blocked,
                "account_blocked": self._account_blocked,
            }

    async def is_tradeable(self) -> bool:
        """Verifica si la cuenta puede operar."""
        status = await self.get_account_status()
        return not any([
            status["trading_blocked"],
            status["account_blocked"],
            status["transfers_blocked"],
        ])

    async def get_snapshot(self) -> AccountSnapshot:
        """Snapshot completo de la cuenta."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return AccountSnapshot(
                equity=self._equity,
                cash=self._cash,
                buying_power=self._buying_power,
                portfolio_value=self._portfolio_value,
                pattern_day_trader=self._pattern_day_trader,
                trading_blocked=self._trading_blocked,
                transfers_blocked=self._transfers_blocked,
                account_blocked=self._account_blocked,
                positions=self._positions.copy(),
                timestamp=time.time(),
            )

    # --- Private methods ---

    async def _auto_sync_loop(self) -> None:
        """Loop de sincronización automática."""
        while True:
            try:
                await asyncio.sleep(self._auto_sync_interval)
                if self._sync_running:
                    await self._sync()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Auto-sync error", extra={"error": str(e)})
                await asyncio.sleep(5)  # Backoff en error

    async def _sync(self) -> None:
        """Sincroniza estado con Alpaca."""
        try:
            # Account info
            account = await self._client.get_account()
            self._equity = float(account.get("equity", 0))
            self._cash = float(account.get("cash", 0))
            self._buying_power = float(account.get("buying_power", 0))
            self._portfolio_value = float(account.get("portfolio_value", 0))
            self._pattern_day_trader = account.get("pattern_day_trader", False)
            self._trading_blocked = account.get("trading_blocked", False)
            self._transfers_blocked = account.get("transfers_blocked", False)
            self._account_blocked = account.get("account_blocked", False)

            # Positions
            positions_data = await self._client.get_positions()
            self._positions = {}
            for pos in positions_data:
                sym = pos["symbol"]
                qty = float(pos["qty"])
                self._positions[sym] = Position(
                    symbol=sym,
                    side=pos["side"],
                    qty=qty,
                    avg_entry_price=float(pos["avg_entry_price"]),
                    market_value=float(pos["market_value"]),
                    cost_basis=float(pos["cost_basis"]),
                    unrealized_pl=float(pos["unrealized_pl"]),
                    unrealized_plpc=float(pos["unrealized_plpc"]),
                    current_price=float(pos["current_price"]),
                    lastday_price=float(pos["lastday_price"]),
                    change_today=float(pos["change_today"]),
                    asset_id=pos["asset_id"],
                    asset_class=pos["asset_class"],
                    exchange=pos["exchange"],
                )

            self._last_sync = time.time()
            logger.debug("Account synced", extra={
                "equity": self._equity,
                "buying_power": self._buying_power,
                "positions": len(self._positions)
            })

        except Exception as e:
            logger.error("Account sync failed", extra={"error": str(e)})
            raise


# Factory
async def create_account(
    client: AlpacaOrderClient,
    auto_sync: bool = True,
) -> AlpacaAccount:
    """Factory para crear y configurar cuenta."""
    account = AlpacaAccount(client)
    if auto_sync:
        await account.start_auto_sync()
    return account