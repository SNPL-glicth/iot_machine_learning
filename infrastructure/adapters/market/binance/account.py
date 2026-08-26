"""BinanceAccount -- Gestión de cuenta y posiciones para Binance Futures."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import defaultdict

from .order_client import BinanceOrderClient

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Posición abierta."""
    symbol: str
    side: str          # "LONG" o "SHORT"
    size: float        # Cantidad (positiva)
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    leverage: int = 1
    margin_type: str = "isolated"
    isolated_margin: float = 0.0
    liquidation_price: float = 0.0
    last_update: float = 0.0


@dataclass
class AccountSnapshot:
    """Snapshot completo de la cuenta."""
    total_equity: float
    available_balance: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_margin_used: float
    available_margin: float
    positions: Dict[str, Any]  # symbol -> Position data
    timestamp: float


class BinanceAccount:
    """
    Gestión de cuenta para Binance Futures.

    Responsabilidades:
    - Sincronización de balances y posiciones
    - Cálculo de PnL (realizado + no realizado)
    - Gestión de margen y apalancamiento
    - Cálculo de equity y margen disponible
    """

    def __init__(
        self,
        client: "BinanceOrderClient",
        symbol: str = "BTCUSDT",
        auto_sync_interval: float = 5.0,
    ):
        self._client = client
        self.symbol = symbol.upper()
        self._auto_sync_interval = auto_sync_interval
        self._sync_task: Optional[asyncio.Task] = None

        # Estado de cuenta
        self._equity: float = 0.0
        self._available_balance: float = 0.0
        self._total_unrealized_pnl: float = 0.0
        self._total_realized_pnl: float = 0.0
        self._total_margin_used: float = 0.0
        self._positions: Dict[str, Any] = {}
        self._last_sync: float = 0.0
        self._sync_running = False
        self._lock = asyncio.Lock()

        logger.info("BinanceAccount initialized", extra={"symbol": symbol})

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
        """Equity total de la cuenta (balance + PnL no realizado)."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return self._equity

    async def get_available_balance(self) -> float:
        """Balance disponible para nuevas posiciones."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return self._available_balance

    async def get_position(self, symbol: Optional[str] = None) -> float:
        """Posición neta (positivo=long, negativo=short)."""
        sym = (symbol or self._client.symbol).upper()
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            pos = self._positions.get(symbol.upper() if symbol else self.symbol.upper())
            if pos:
                return pos.get("positionAmt", 0.0)
            return 0.0

    async def get_position_details(self, symbol: Optional[str] = None) -> Optional[Dict]:
        """Detalles completos de posición."""
        sym = (symbol or self.symbol).upper()
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return self._positions.get(sym)

    async def get_unrealized_pnl(self, symbol: Optional[str] = None) -> float:
        """PnL no realizado de una posición."""
        sym = (symbol or self.symbol).upper()
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            pos = self._positions.get(sym)
            if pos:
                return float(pos.get("unRealizedProfit", 0.0))
            return 0.0

    async def get_total_unrealized_pnl(self) -> float:
        """PnL no realizado total."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return self._total_unrealized_pnl

    async def get_total_realized_pnl(self) -> float:
        """PnL realizado total."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return self._total_realized_pnl

    async def get_margin_info(self) -> Dict[str, float]:
        """Información de margen."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return {
                "total_margin_used": self._total_margin_used,
                "available_margin": self._available_balance,
                "total_equity": self._equity,
            }

    async def get_snapshot(self) -> Dict:
        """Snapshot completo de la cuenta."""
        async with self._lock:
            if time.time() - self._last_sync > 1.0:
                await self._sync()
            return {
                "equity": self._equity,
                "available_balance": self._available_balance,
                "total_unrealized_pnl": self._total_unrealized_pnl,
                "total_realized_pnl": self._total_realized_pnl,
                "total_margin_used": self._total_margin_used,
                "available_margin": self._available_balance,
                "positions": self._positions,
                "timestamp": time.time(),
            }

    # --- Private methods ---

    async def _auto_sync_loop(self) -> None:
        """Loop de sincronización automática."""
        while True:
            try:
                await asyncio.sleep(self._auto_sync_interval)
                if hasattr(self, '_client') and hasattr(self, '_sync_running') and self._sync_running:
                    await self._sync()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Auto-sync error", extra={"error": str(e)})
                await asyncio.sleep(5)  # Backoff en error

    async def _sync(self) -> None:
        """Sincroniza estado con Binance."""
        try:
            # Account info
            account = await self._client.get_account_info()
            self._equity = float(account.get("totalWalletBalance", 0))
            self._available_balance = float(account.get("availableBalance", 0))
            self._total_unrealized_pnl = float(account.get("totalUnrealizedProfit", 0))
            self._total_realized_pnl = 0.0  # No directamente disponible
            self._total_margin_used = float(account.get("totalMarginBalance", 0)) - self._available_balance

            # Positions
            positions = await self._client.get_position_risk()
            self._positions = {}
            for pos in positions:
                sym = pos["symbol"]
                amt = float(pos["positionAmt"])
                if amt != 0:
                    self._positions[sym] = {
                        "symbol": pos["symbol"],
                        "positionAmt": amt,
                        "entryPrice": float(pos["entryPrice"]),
                        "markPrice": float(pos["markPrice"]),
                        "unRealizedProfit": float(pos["unRealizedProfit"]),
                        "leverage": int(pos.get("leverage", 1)),
                        "marginType": pos.get("marginType", "isolated"),
                        "isolatedMargin": float(pos.get("isolatedMargin", 0)),
                        "liquidationPrice": float(pos.get("liquidationPrice", 0)),
                    }

            self._last_sync = time.time()
            logger.debug("Account synced", extra={"equity": self._equity, "positions": len(self._positions)})

        except Exception as e:
            logger.error("Account sync failed", extra={"error": str(e)})
            raise

    # --- Helpers ---

    async def get_leverage(self, symbol: str) -> int:
        """Obtiene apalancamiento actual."""
        pos = await self.get_position_details(symbol)
        return pos.get("leverage", 1) if pos else 1

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Establece apalancamiento (1-125 para BTCUSDT)."""
        try:
            await self._client._request(
                "POST", "/fapi/v1/leverage",
                {"symbol": symbol.upper(), "leverage": leverage},
                signed=True, weight=1
            )
            return True
        except Exception as e:
            logger.error("Failed to set leverage", extra={"error": str(e)})
            return False

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """Establece tipo de margen (ISOLATED/CROSSED)."""
        try:
            await self._client._request(
                "POST", "/fapi/v1/marginType",
                {"symbol": symbol.upper(), "marginType": margin_type.upper()},
                signed=True, weight=1
            )
            return True
        except Exception as e:
            logger.error("Failed to set margin type", extra={"error": str(e)})
            return False


# Factory
async def create_account(
    client: "BinanceOrderClient",
    symbol: str = "BTCUSDT",
    auto_sync: bool = True,
) -> "BinanceAccount":
    """Factory para crear y configurar cuenta."""
    account = BinanceAccount(client, symbol)
    if auto_sync:
        await account.start_auto_sync()
    return account