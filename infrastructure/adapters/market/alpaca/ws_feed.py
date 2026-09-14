"""AlpacaWSFeed — Feed asíncrono de mercado para Alpaca Paper Trading."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

from iot_machine_learning.domain.entities.market.observations import MarketObservation
from iot_machine_learning.infrastructure.adapters.market.alpaca.connection_manager import (
    connect_and_auth,
    perform_reconnect,
    start_feed_tasks,
    subscribe_streams,
)
from iot_machine_learning.infrastructure.adapters.market.alpaca.feed_models import (
    ConnectionState,
    FeedStats,
)

logger = logging.getLogger(__name__)


class AlpacaWSFeed:
    """Feed asíncrono de Alpaca para trading event-driven multiplexado."""

    order_book: Any = None
    order_book_metrics: Any = None

    def __init__(
        self, symbol: str | list[str] = "SPY", *, api_key: str, api_secret: str, data_feed: str = "iex",
        include_trades: bool = True, include_quotes: bool = True, include_bars: bool = True,
        bar_interval: str = "1Min", max_queue_size: int = 10000,
        on_observation: Callable[[MarketObservation], None] | None = None,
        on_metrics: Callable[[dict], None] | None = None,
        on_state_change: Callable[[str, str], None] | None = None,
    ):
        self.symbols = [symbol.upper()] if isinstance(symbol, str) else [s.upper() for s in symbol]
        self.symbol = self.symbols[0] if self.symbols else "SPY"
        self.api_key, self.api_secret, self.data_feed, self.bar_interval = api_key, api_secret, data_feed, bar_interval
        self.streams = [s for s, inc in [("trades", include_trades), ("quotes", include_quotes), ("bars", include_bars)] if inc]
        self.ws_url = f"wss://stream.data.alpaca.markets/v2/{data_feed}"
        self._running, self._connected = False, False
        self._state: str = ConnectionState.DISCONNECTED
        self._ws: Any | None = None
        self._reconnect_count, self._latest_quote = 0, None
        self._latest_quotes: dict[str, Any] = {}
        self._emit_task: asyncio.Task[Any] | None = None
        self._health_task: asyncio.Task[Any] | None = None
        self._ping_task: asyncio.Task[Any] | None = None
        self._obs_queue: asyncio.Queue[MarketObservation] = asyncio.Queue(maxsize=max_queue_size)
        self.on_observation, self.on_metrics, self.on_state_change = on_observation, on_metrics, on_state_change
        self.stats = FeedStats()
        logger.info("AlpacaWSFeed initialized", extra={"symbols": self.symbols, "streams": self.streams})

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def state(self) -> str:
        return self._state

    @property
    def best_bid(self) -> float | None:
        return float(self._latest_quote.bid) if getattr(self._latest_quote, "bid", None) is not None else None

    @property
    def best_ask(self) -> float | None:
        return float(self._latest_quote.ask) if getattr(self._latest_quote, "ask", None) is not None else None

    @property
    def mid_price(self) -> float | None:
        return self.get_mid_price(self.symbol)

    def get_mid_price(self, sym: str | None = None) -> float | None:
        q = self._latest_quotes.get(sym.upper()) if sym else self._latest_quote
        b, a = getattr(q, "bid", None), getattr(q, "ask", None)
        return (float(b) + float(a)) / 2.0 if (b is not None and a is not None) else None

    @property
    def spread(self) -> float | None:
        return (self.best_ask - self.best_bid) if (self.best_bid is not None and self.best_ask is not None) else None

    def _set_state(self, new_state: str) -> None:
        old, self._state = self._state, new_state
        if self.on_state_change and old != new_state:
            try:
                self.on_state_change(old, new_state)
            except Exception as e:
                logger.error("State callback error", extra={"error": str(e)})

    def _on_quote_received(self, obs: Any) -> None:
        self._latest_quote = obs
        if hasattr(obs, "symbol"):
            self._latest_quotes[obs.symbol] = obs

    async def connect(self) -> None:
        """Inicia conexión WebSocket y tareas de fondo."""
        if self._running:
            return
        self._running = True
        self._set_state(ConnectionState.CONNECTING)
        try:
            self._ws = await connect_and_auth(self.ws_url, self.api_key, self.api_secret, self.symbol)
            self._connected = True
            self._set_state(ConnectionState.CONNECTED)
            await subscribe_streams(self._ws, self.symbols, self.streams)
            self._emit_task, self._ping_task, self._health_task = start_feed_tasks(
                lambda: self._ws, lambda: self._running, self._reconnect,
                self.on_observation, self._obs_queue, self.symbol, self.bar_interval,
                self.stats, self._on_quote_received, self.on_metrics,
            )
            logger.info("AlpacaWSFeed connected", extra={"symbols": self.symbols})
        except Exception as e:
            self._connected = False
            self._set_state(ConnectionState.ERROR)
            logger.error("AlpacaWSFeed connection failed", extra={"symbols": self.symbols, "error": str(e)})
            raise

    async def disconnect(self) -> None:
        """Desconecta el feed limpiamente."""
        self._running = False
        for t in (self._emit_task, self._health_task, self._ping_task):
            if t:
                t.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        self._connected = False
        self._set_state(ConnectionState.DISCONNECTED)
        logger.info("AlpacaWSFeed disconnected", extra={"symbols": self.symbols})

    async def _reconnect(self) -> None:
        self._connected = False
        self._set_state(ConnectionState.RECONNECTING)
        self._reconnect_count += 1
        try:
            self._ws = await perform_reconnect(
                self.ws_url, self.api_key, self.api_secret, self.symbol, self.symbols, self.streams, self._reconnect_count
            )
            self._connected = True
            self._set_state(ConnectionState.CONNECTED)
            self.stats.reconnects += 1
            logger.info("Reconnected successfully", extra={"symbols": self.symbols})
        except Exception as e:
            logger.error("Reconnection failed", extra={"symbols": self.symbols, "error": str(e)})
            if self._running:
                await self._reconnect()

    def __aiter__(self) -> AsyncGenerator[MarketObservation, None]:
        return self.iter_observations()

    async def iter_observations(self) -> AsyncGenerator[MarketObservation, None]:
        while self._running:
            try:
                yield await asyncio.wait_for(self._obs_queue.get(), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                if not self._running:
                    break

    async def get_next_observation(self, timeout: float | None = None) -> MarketObservation | None:
        try:
            return await (self._obs_queue.get() if timeout is None else asyncio.wait_for(self._obs_queue.get(), timeout=timeout))
        except asyncio.TimeoutError:
            return None

    def get_stats(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol, "symbols": self.symbols, "running": self._running,
            "connected": self._connected, "state": self._state,
            "feed_stats": self.stats.to_dict(), "reconnect_count": self._reconnect_count,
        }
#princes trainner

AlpacaLiveFeed = AlpacaWSFeed
