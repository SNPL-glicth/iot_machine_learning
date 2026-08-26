"""BinanceWSFeed — Feed asíncrono de mercado para Binance (L2 Depth + Trades).

Implementa FeedProtocol asíncrono para el loop event-driven:
- WebSocket L2 Depth @100ms + aggTrade + bookTicker
- Sincronización order book: snapshot REST + deltas WS
- Emite MarketObservation (Candle, Quote, Trade, OrderBookSnapshot)
- Métricas de latencia feed→feature
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Set, Any
from collections import deque

from iot_machine_learning.domain.entities.market.observations import (
    MarketObservation, Candle, Quote, Trade, OrderBookSnapshot
)
from iot_machine_learning.domain.entities.market import DataStatus

from .ws_client import BinanceWSClient, create_market_streams, ConnectionState
from .order_book_state import OrderBookL2, OrderBookMetrics

logger = logging.getLogger(__name__)


@dataclass
class FeedStats:
    """Estadísticas del feed para monitoreo."""
    events_received: int = 0
    events_emitted: int = 0
    depth_updates: int = 0
    trades_received: int = 0
    quotes_received: int = 0
    candles_received: int = 0
    gaps_detected: int = 0
    reconnects: int = 0
    last_event_time: float = 0.0
    last_emitted_time: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "events_received": self.events_received,
            "events_emitted": self.events_emitted,
            "depth_updates": self.depth_updates,
            "trades_received": self.trades_received,
            "quotes_received": self.quotes_received,
            "candles_received": self.candles_received,
            "gaps_detected": self.gaps_detected,
            "reconnects": self.reconnects,
            "avg_latency_ms": self.avg_latency_ms,
            "max_latency_ms": self.max_latency_ms,
        }


class BinanceWSFeed:
    """
    Feed asíncrono de Binance para trading event-driven.
    
    Características:
    - Conexión WebSocket gestionada (reconexión, ping/pong)
    - Order book L2 sincronizado (snapshot REST + deltas WS)
    - Emite observaciones tipadas (Candle, Quote, Trade, OrderBookSnapshot)
    - Detección de gaps y estado de conexión
    - Métricas de latencia feed→observación
    """
    
    def __init__(
        self,
        symbol: str,
        *,
        testnet: bool = False,
        depth_speed: str = "100ms",
        include_trades: bool = True,
        include_book_ticker: bool = True,
        include_kline: bool = False,
        kline_interval: str = "1m",
        max_queue_size: int = 10000,
        snapshot_interval_sec: float = 30.0,
        on_observation: Optional[callable] = None,
        on_metrics: Optional[callable] = None,
        on_state_change: Optional[callable] = None,
    ):
        """
        Args:
            symbol: Símbolo (ej. "BTCUSDT")
            testnet: Usar Binance Futures Testnet
            depth_speed: "100ms" o "1000ms" para depth stream
            include_trades: Incluir stream aggTrade
            include_book_ticker: Incluir stream bookTicker
            include_kline: Incluir stream kline
            kline_interval: Intervalo para kline (ej. "1m", "5m")
            max_queue_size: Buffer interno
            snapshot_interval_sec: Intervalo resync order book
            on_observation: Callback por observación emitida
            on_metrics: Callback por actualización de métricas
            on_state_change: Callback cambio de estado
        """
        self.symbol = symbol.upper()
        self.testnet = testnet
        self.depth_speed = depth_speed
        
        # Streams a suscribir
        streams = create_market_streams(symbol, depth_speed)
        if not include_trades:
            streams = [s for s in streams if "aggTrade" not in s]
        if not include_book_ticker:
            streams = [s for s in streams if "bookTicker" not in s]
        if include_kline:
            streams.append(f"{symbol.lower()}@kline_{kline_interval}")
        
        # Cliente WebSocket
        self._ws_client = BinanceWSClient(
            symbol=symbol,
            streams=streams,
            testnet=testnet,
            on_message=self._on_raw_message,
            on_state_change=self._on_ws_state_change,
        )
        
        # Order book L2
        self._order_book = OrderBookL2(symbol)
        self._order_book.on_update = self._on_order_book_update
        self._order_book.on_metrics_update = self._on_metrics_update
        
        # Estado
        self._running = False
        self._initialized = False
        self._last_sequence: Dict[str, int] = defaultdict(int)
        
        # Buffer de observaciones listas para emitir
        self._obs_queue: asyncio.Queue[MarketObservation] = asyncio.Queue(maxsize=10000)
        
        # Callbacks
        self.on_observation = on_observation
        self.on_metrics = on_metrics
        self.on_state_change = on_state_change
        
        # Estadísticas
        self.stats = FeedStats()
        
        # Configuración order book
        self._order_book.snapshot_interval_sec = 30.0
        
        logger.info(
            "BinanceWSFeed initialized",
            extra={"symbol": self.symbol, "testnet": testnet, "depth_speed": depth_speed},
        )
    
    @property
    def symbol(self) -> str:
        return self._ws_client.symbol
    
    @property
    def is_connected(self) -> bool:
        return self._ws_client.is_connected
    
    @property
    def state(self) -> ConnectionState:
        return self._ws_client.state
    
    @property
    def order_book(self) -> OrderBookL2:
        return self._order_book
    
    @property
    def order_book_metrics(self) -> Optional[OrderBookMetrics]:
        return self._order_book.metrics
    
    @property
    def best_bid(self) -> Optional[float]:
        return self._order_book.best_bid
    
    @property
    def best_ask(self) -> Optional[float]:
        return self._order_book.best_ask
    
    @property
    def mid_price(self) -> Optional[float]:
        return self._order_book.mid_price
    
    @property
    def spread(self) -> Optional[float]:
        return self._order_book.spread
    
    async def connect(self) -> None:
        """Inicia conexión WebSocket y sincronización inicial."""
        if self._running:
            logger.warning("Feed already running")
            return
        
        self._running = True
        
        # Conectar WebSocket
        await self._ws_client.connect()
        
        # Sincronización inicial del order book
        await self._initial_sync()
        
        # Iniciar task de emisión de observaciones
        self._emit_task = asyncio.create_task(self._emit_loop())
        self._health_task = asyncio.create_task(self._health_loop())
        
        logger.info("BinanceWSFeed connected and syncing", extra={"symbol": self.symbol})
    
    async def disconnect(self) -> None:
        """Desconecta limpiamente."""
        self._running = False
        
        if hasattr(self, '_emit_task'):
            self._emit_task.cancel()
            try:
                await self._emit_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(self, '_health_task'):
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        await self._ws_client.disconnect()
        logger.info("BinanceWSFeed disconnected", extra={"symbol": self.symbol})
    
    async def _initial_sync(self) -> None:
        """Sincronización inicial: snapshot REST + buffer deltas."""
        logger.info("Starting initial order book sync", extra={"symbol": self.symbol})
        
        # 1. Marcar inicio de snapshot
        self._order_book.start_snapshot_sync()
        
        # 2. Fetch snapshot REST
        snapshot = await self._ws_client.get_snapshot_rest()
        
        # 3. Aplicar snapshot (procesa buffer automáticamente)
        success = self._order_book.apply_snapshot(snapshot)
        
        if not success:
            raise RuntimeError("Initial order book sync failed")
        
        self._initialized = True
        logger.info(
            "Initial order book sync complete",
            extra={"symbol": self.symbol, "update_id": self._order_book.last_update_id},
        )
    
    async def _emit_loop(self) -> None:
        """Loop que procesa mensajes WS y emite observaciones."""
        try:
            async for raw_msg in self._ws_client.iter_messages():
                if not self._running:
                    break
                
                self.stats.events_received += 1
                receive_time = time.time()
                
                # Procesar mensaje y convertir a observaciones
                observations = self._process_raw_message(raw_msg, receive_time)
                
                for obs in observations:
                    self.stats.events_emitted += 1
                    self.stats.last_emitted_time = time.time()
                    
                    # Calcular latencia feed→obs
                    latency_ms = (time.time() - obs.timestamp) * 1000
                    self.stats.avg_latency_ms = (
                        self.stats.avg_latency_ms * 0.99 + latency_ms * 0.01
                    )
                    self.stats.max_latency_ms = max(self.stats.max_latency_ms, latency_ms)
                    
                    # Enqueue para consumidor
                    try:
                        self._obs_queue.put_nowait(obs)
                    except asyncio.QueueFull:
                        logger.warning("Observation queue full, dropping", extra={"symbol": self.symbol})
                    
                    # Callback opcional
                    if self.on_observation:
                        try:
                            self.on_observation(obs)
                        except Exception as e:
                            logger.error("Error in on_observation callback", extra={"error": str(e)})
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Emit loop error", extra={"symbol": self.symbol, "error": str(e)})
    
    def _process_raw_message(self, msg: Dict, receive_time: float) -> List[MarketObservation]:
        """Convierte mensaje raw WS a lista de MarketObservation."""
        observations = []
        
        # Binance combined stream format: {"stream": "...", "data": {...}}
        if "stream" in msg and "data" in msg:
            stream_name = msg["stream"]
            data = msg["data"]
            event_type = stream_name.split("@")[-1]
        else:
            # Single stream format
            data = msg
            event_type = data.get("e", "unknown")
        
        # Añadir timestamp de recepción
        data["_received_at"] = receive_time
        
        try:
            if "depth" in event_type or "depthUpdate" == data.get("e"):
                # Depth update - actualizar order book
                self.stats.depth_updates += 1
                self._order_book.apply_delta(data)
                # Emitir OrderBookSnapshot periódicamente o en cambios significativos
                obs = self._create_orderbook_snapshot(receive_time)
                if obs:
                    observations.append(obs)
                    
            elif data.get("e") == "aggTrade":
                # Trade agregado
                self.stats.trades_received += 1
                obs = self._create_trade(data, receive_time)
                observations.append(obs)
                
            elif data.get("e") == "bookTicker" or "bookTicker" in str(data):
                # Best bid/ask update
                self.stats.quotes_received += 1
                obs = self._create_quote(data, receive_time)
                observations.append(obs)
                
            elif "kline" in event_type or data.get("e") == "kline":
                # Kline/candle
                self.stats.candles_received += 1
                obs = self._create_candle(data, receive_time)
                observations.append(obs)
                
        except Exception as e:
            logger.error("Error processing message", extra={"error": str(e), "msg_type": event_type})
        
        return observations
    
    def _create_trade(self, data: Dict, receive_time: float) -> Trade:
        """Crea Trade desde aggTrade."""
        return Trade(
            symbol=self.symbol,
            timestamp=data["T"] / 1000.0,  # Binance usa ms
            data_status=DataStatus.LIVE,
            source_provider="binance",
            price=float(data["p"]),
            size=float(data["q"]),
            trade_id=str(data["a"]),
            taker_side="sell" if data.get("m", False) else "buy",  # m = isBuyerMaker
        )
    
    def _create_quote(self, data: Dict, receive_time: float) -> Quote:
        """Crea Quote desde bookTicker."""
        return Quote(
            symbol=self.symbol,
            timestamp=receive_time,
            data_status=DataStatus.LIVE,
            source_provider="binance",
            bid=float(data["b"]),
            bid_size=float(data["B"]),
            ask=float(data["a"]),
            ask_size=float(data["A"]),
        )
    
    def _create_candle(self, data: Dict, receive_time: float) -> Candle:
        """Crea Candle desde kline."""
        k = data.get("k", data)
        return Candle(
            symbol=self.symbol,
            timestamp=k["t"] / 1000.0,
            data_status=DataStatus.LIVE if not k.get("x") else DataStatus.CLOSED,
            source_provider="binance",
            open=float(k["o"]),
            high=float(k["h"]),
            low=float(k["l"]),
            close=float(k["c"]),
            volume=float(k["v"]),
            interval_seconds=self._interval_to_seconds(k.get("i", "1m")),
            vwap=float(k.get("Q", 0)) / float(k["v"]) if float(k.get("v", 0)) > 0 else None,
            trade_count=int(k.get("n", 0)),
        )
    
    def _interval_to_seconds(self, interval: str) -> int:
        """Convierte intervalo Binance a segundos."""
        unit = interval[-1]
        value = int(interval[:-1])
        multipliers = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
        return value * multipliers.get(unit, 60)
    
    def _create_orderbook_snapshot(self, receive_time: float) -> Optional[OrderBookSnapshot]:
        """Crea OrderBookSnapshot desde estado actual."""
        if not self._order_book.is_initialized:
            return None
        
        bids, asks = self._order_book.get_top_levels(20)
        
        return OrderBookSnapshot(
            symbol=self.symbol,
            timestamp=receive_time,
            data_status=DataStatus.LIVE,
            source_provider="binance",
            bids=tuple(bids),
            asks=tuple(asks),
            reset=False,
        )
    
    def _on_raw_message(self, msg: Dict) -> None:
        """Callback para mensaje raw (para métricas/debug)."""
        pass
    
    def _on_ws_state_change(self, old: ConnectionState, new: ConnectionState) -> None:
        """Callback cambio de estado WS."""
        if new == ConnectionState.CONNECTED and self._ws_client.metrics.reconnect_count > 0:
            self.stats.reconnects += 1
            logger.info("WS reconnected", extra={"symbol": self.symbol})
        
        if self.on_state_change:
            try:
                self.on_state_change(old, new)
            except Exception as e:
                logger.error("Error in state change callback", extra={"error": str(e)})
    
    def _on_order_book_update(self, ob: OrderBookL2) -> None:
        """Callback actualización order book."""
        # Detectar gaps en update_id si es posible
        pass
    
    def _on_metrics_update(self, metrics: OrderBookMetrics) -> None:
        """Callback métricas order book."""
        if self.on_metrics:
            try:
                self.on_metrics(metrics)
            except Exception as e:
                logger.error("Error in metrics callback", extra={"error": str(e)})
    
    async def _health_loop(self) -> None:
        """Loop de salud: resync periódico + métricas."""
        while self._running:
            await asyncio.sleep(10)  # Check cada 10s
            
            if not self._running:
                break
            
            # Resync periódico
            if self._order_book.needs_resync():
                logger.info("Periodic order book resync", extra={"symbol": self.symbol})
                try:
                    self._order_book.start_snapshot_sync()
                    snapshot = await self._ws_client.get_snapshot_rest()
                    self._order_book.apply_snapshot(snapshot)
                except Exception as e:
                    logger.error("Periodic resync failed", extra={"symbol": self.symbol, "error": str(e)})
            
            # Callback métricas
            if self.on_metrics:
                try:
                    self.on_metrics(self.stats.to_dict())
                except Exception as e:
                    logger.error("Error in health metrics callback", extra={"error": str(e)})
    
    # Async iterator protocol
    def __aiter__(self) -> AsyncGenerator[MarketObservation, None]:
        return self.iter_observations()
    
    async def iter_observations(self) -> AsyncGenerator[MarketObservation, None]:
        """Iterador asíncrono sobre observaciones de mercado."""
        while self._running:
            try:
                obs = await asyncio.wait_for(self._obs_queue.get(), timeout=1.0)
                yield obs
            except asyncio.TimeoutError:
                if not self._running:
                    break
                continue
            except Exception as e:
                logger.error("Error in observation iterator", extra={"error": str(e)})
                if not self._running:
                    break
    
    async def get_next_observation(self, timeout: Optional[float] = None) -> Optional[MarketObservation]:
        """Obtiene próxima observación (para uso síncrono)."""
        try:
            if timeout is None:
                return await self._obs_queue.get()
            return await asyncio.wait_for(self._obs_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    def get_stats(self) -> Dict:
        """Estadísticas completas del feed."""
        ws_info = self._ws_client.get_connection_info()
        ob_metrics = self._order_book.metrics
        
        return {
            "symbol": self.symbol,
            "running": self._running,
            "initialized": self._initialized,
            "feed_stats": self.stats.to_dict(),
            "ws_client": ws_info,
            "order_book_metrics": ob_metrics.to_dict() if ob_metrics else None,
        }


# Alias para compatibilidad
BinanceLiveFeed = BinanceWSFeed