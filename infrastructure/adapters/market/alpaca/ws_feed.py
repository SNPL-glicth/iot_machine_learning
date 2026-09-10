"""AlpacaWSFeed — Feed asíncrono de mercado para Alpaca Paper Trading."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from iot_machine_learning.domain.entities.market.observations import MarketObservation
from iot_machine_learning.infrastructure.adapters.market.alpaca.connection_manager import (
    connect_and_auth, get_reconnect_delay, subscribe_streams,
)
from iot_machine_learning.infrastructure.adapters.market.alpaca.feed_models import ConnectionState, FeedStats
from iot_machine_learning.infrastructure.adapters.market.alpaca.message_parser import parse_raw_message

logger = logging.getLogger(__name__)


class AlpacaWSFeed:
    """Feed asíncrono de Alpaca para trading event-driven."""

    def __init__(
        self, symbol: str, *, api_key: str, api_secret: str, data_feed: str = "iex",
        include_trades: bool = True, include_quotes: bool = True, include_bars: bool = True,
        bar_interval: str = "1Min", max_queue_size: int = 10000,
        on_observation: Optional[Callable[[MarketObservation], None]] = None,
        on_metrics: Optional[Callable[[dict], None]] = None,
        on_state_change: Optional[Callable[[str, str], None]] = None,
    ):
        self.symbol, self.api_key, self.api_secret = symbol.upper(), api_key, api_secret
        self.data_feed, self.bar_interval = data_feed, bar_interval
        self.streams = [s for s, inc in [("trades", include_trades), ("quotes", include_quotes), ("bars", include_bars)] if inc]
        self.ws_url = f"wss://stream.data.alpaca.markets/v2/{data_feed}"
        self._running, self._connected = False, False
        self._state, self._ws, self._reconnect_count, self._last_ping = ConnectionState.DISCONNECTED, None, 0, 0.0
        self._obs_queue: asyncio.Queue[MarketObservation] = asyncio.Queue(maxsize=max_queue_size)
        self.on_observation, self.on_metrics, self.on_state_change = on_observation, on_metrics, on_state_change
        self.stats = FeedStats()
        self._emit_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        logger.info("AlpacaWSFeed initialized", extra={"symbol": self.symbol, "streams": self.streams})

    @property
    def is_connected(self) -> bool: return self._connected
    @property
    def state(self) -> str: return self._state
    @property
    def order_book(self) -> None: return None
    @property
    def order_book_metrics(self) -> None: return None
    @property
    def best_bid(self) -> Optional[float]: return None
    @property
    def best_ask(self) -> Optional[float]: return None
    @property
    def mid_price(self) -> Optional[float]: return None
    @property
    def spread(self) -> Optional[float]: return None

    def _set_state(self, new_state: str) -> None:
        old, self._state = self._state, new_state
        if self.on_state_change and old != new_state:
            try: self.on_state_change(old, new_state)
            except Exception as e: logger.error("Error in state change callback", extra={"error": str(e)})

    async def connect(self) -> None:
        """Inicia conexión WebSocket y tareas de fondo."""
        if self._running: return
        self._running = True
        self._set_state(ConnectionState.CONNECTING)
        try:
            self._ws = await connect_and_auth(self.ws_url, self.api_key, self.api_secret, self.symbol)
            self._connected = True; self._set_state(ConnectionState.CONNECTED)
            await subscribe_streams(self._ws, self.symbol, self.streams)
            self._emit_task = asyncio.create_task(self._emit_loop())
            self._health_task = asyncio.create_task(self._health_loop())
            self._ping_task = asyncio.create_task(self._ping_loop())
            logger.info("AlpacaWSFeed connected", extra={"symbol": self.symbol})
        except Exception as e:
            self._connected = False; self._set_state(ConnectionState.ERROR)
            logger.error("AlpacaWSFeed connection failed", extra={"symbol": self.symbol, "error": str(e)})
            raise

    async def disconnect(self) -> None:
        """Desconecta el feed limpiamente."""
        self._running = False
        for t in (self._emit_task, self._health_task, self._ping_task):
            if t: t.cancel()
        if self._ws:
            try: await self._ws.close()
            except Exception: pass
        self._connected = False; self._set_state(ConnectionState.DISCONNECTED)
        logger.info("AlpacaWSFeed disconnected", extra={"symbol": self.symbol})

    async def _emit_loop(self) -> None:
        try:
            while self._running and self._ws:
                try:
                    msg = await asyncio.wait_for(self._ws.recv(), timeout=1.0)
                    self.stats.events_received += 1
                    for obs in parse_raw_message(msg, self.symbol, self.bar_interval, time.time(), self.stats):
                        self.stats.events_emitted += 1; self.stats.last_emitted_time = time.time()
                        lat_ms = (time.time() - obs.timestamp) * 1000
                        self.stats.avg_latency_ms = self.stats.avg_latency_ms * 0.99 + lat_ms * 0.01
                        self.stats.max_latency_ms = max(self.stats.max_latency_ms, lat_ms)
                        try: self._obs_queue.put_nowait(obs)
                        except asyncio.QueueFull: logger.warning("Queue full, drop", extra={"symbol": self.symbol})
                        if self.on_observation:
                            try: self.on_observation(obs)
                            except Exception as e: logger.error("Callback error", extra={"error": str(e)})
                except asyncio.TimeoutError: continue
                except Exception as e:
                    logger.error("Emit error", extra={"symbol": self.symbol, "error": str(e)})
                    if self._running: await self._reconnect()
        except asyncio.CancelledError: pass

    async def _reconnect(self) -> None:
        self._connected = False; self._set_state(ConnectionState.RECONNECTING)
        self._reconnect_count += 1
        delay = get_reconnect_delay(self._reconnect_count)
        logger.info(f"Reconnecting in {delay:.1f}s...", extra={"symbol": self.symbol, "attempt": self._reconnect_count})
        await asyncio.sleep(delay)
        if self._running:
            try:
                self._ws = await connect_and_auth(self.ws_url, self.api_key, self.api_secret, self.symbol)
                await subscribe_streams(self._ws, self.symbol, self.streams)
                self._connected = True; self._set_state(ConnectionState.CONNECTED)
                self.stats.reconnects += 1
                logger.info("Reconnected successfully", extra={"symbol": self.symbol})
            except Exception as e:
                logger.error("Reconnection failed", extra={"symbol": self.symbol, "error": str(e)})
                await self._reconnect()

    async def _ping_loop(self) -> None:
        while self._running:
            await asyncio.sleep(20)
            if self._ws and self._connected:
                try: await self._ws.ping(); self._last_ping = time.time()
                except Exception as e:
                    logger.warning("Ping failed", extra={"error": str(e)}); await self._reconnect()

    async def _health_loop(self) -> None:
        while self._running:
            await asyncio.sleep(10)
            if not self._running: break
            if self.on_metrics:
                try: self.on_metrics(self.stats.to_dict())
                except Exception as e: logger.error("Metrics callback error", extra={"error": str(e)})

    def __aiter__(self) -> AsyncGenerator[MarketObservation, None]: return self.iter_observations()

    async def iter_observations(self) -> AsyncGenerator[MarketObservation, None]:
        while self._running:
            try: yield await asyncio.wait_for(self._obs_queue.get(), timeout=1.0)
            except (asyncio.TimeoutError, Exception):
                if not self._running: break

    async def get_next_observation(self, timeout: Optional[float] = None) -> Optional[MarketObservation]:
        try: return await (self._obs_queue.get() if timeout is None else asyncio.wait_for(self._obs_queue.get(), timeout=timeout))
        except asyncio.TimeoutError: return None

    def get_stats(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol, "running": self._running, "connected": self._connected,
            "state": self._state, "feed_stats": self.stats.to_dict(), "reconnect_count": self._reconnect_count,
        }


AlpacaLiveFeed = AlpacaWSFeed