"""AlpacaWSFeed — Feed asíncrono de mercado para Alpaca Paper Trading.

Implementa feed asíncrono para Alpaca:
- WebSocket para trades, quotes, bars (v2 API)
- Emite MarketObservation (Trade, Quote, Candle)
- Reconnection, ping/pong handling
- Métricas de latencia feed→feature
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import AsyncGenerator, Dict, List, Optional, Any
from collections import deque

from iot_machine_learning.domain.entities.market.observations import (
    MarketObservation, Candle, Quote, Trade, OrderBookSnapshot
)
from iot_machine_learning.domain.entities.market import DataStatus

logger = logging.getLogger(__name__)


@dataclass
class FeedStats:
    """Estadísticas del feed para monitoreo."""
    events_received: int = 0
    events_emitted: int = 0
    trades_received: int = 0
    quotes_received: int = 0
    quotes_stale_dropped: int = 0
    bars_received: int = 0
    reconnects: int = 0
    last_event_time: float = 0.0
    last_emitted_time: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "events_received": self.events_received,
            "events_emitted": self.events_emitted,
            "trades_received": self.trades_received,
            "quotes_received": self.quotes_received,
            "quotes_stale_dropped": self.quotes_stale_dropped,
            "bars_received": self.bars_received,
            "reconnects": self.reconnects,
            "avg_latency_ms": self.avg_latency_ms,
            "max_latency_ms": self.max_latency_ms,
        }


class ConnectionState:
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class AlpacaWSFeed:
    """
    Feed asíncrono de Alpaca para trading event-driven.
    
    Características:
    - Conexión WebSocket gestionada (reconexión, ping/pong)
    - Streams: trades, quotes, bars (Alpaca Data API v2)
    - Emite observaciones tipadas (Trade, Quote, Candle)
    - Detección de gaps y estado de conexión
    - Métricas de latencia feed→observación
    """
    
    def __init__(
        self,
        symbol: str,
        *,
        api_key: str,
        api_secret: str,
        data_feed: str = "iex",  # "iex" or "sip"
        include_trades: bool = True,
        include_quotes: bool = True,
        include_bars: bool = True,
        bar_interval: str = "1Min",
        max_queue_size: int = 10000,
        on_observation: Optional[callable] = None,
        on_metrics: Optional[callable] = None,
        on_state_change: Optional[callable] = None,
    ):
        """
        Args:
            symbol: Símbolo (ej. "SPY", "AAPL")
            api_key: Alpaca API Key
            api_secret: Alpaca Secret Key
            data_feed: "iex" o "sip"
            include_trades: Incluir stream trades
            include_quotes: Incluir stream quotes
            include_bars: Incluir stream bars
            bar_interval: Intervalo para bars (ej. "1Min", "5Min")
            max_queue_size: Buffer interno
            on_observation: Callback por observación emitida
            on_metrics: Callback por actualización de métricas
            on_state_change: Callback cambio de estado
        """
        self.symbol = symbol.upper()
        self.api_key = api_key
        self.api_secret = api_secret
        self.data_feed = data_feed
        self.bar_interval = bar_interval
        
        # Build stream subscriptions
        self.streams = []
        if include_trades:
            self.streams.append("trades")
        if include_quotes:
            self.streams.append("quotes")
        if include_bars:
            self.streams.append(f"bars")
        
        # WebSocket URL (Alpaca Data API v2)
        self.ws_url = f"wss://stream.data.alpaca.markets/v2/{data_feed}"
        
        # Estado
        self._running = False
        self._connected = False
        self._state = ConnectionState.DISCONNECTED
        self._ws: Optional[Any] = None
        self._reconnect_count = 0
        self._last_ping = 0.0
        
        # Buffer de observaciones
        self._obs_queue: asyncio.Queue[MarketObservation] = asyncio.Queue(maxsize=max_queue_size)
        
        # Callbacks
        self.on_observation = on_observation
        self.on_metrics = on_metrics
        self.on_state_change = on_state_change
        
        # Estadísticas
        self.stats = FeedStats()
        
        # Tasks
        self._emit_task: Optional[asyncio.Task] = None
        self._health_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
        
        logger.info(
            "AlpacaWSFeed initialized",
            extra={
                "symbol": self.symbol,
                "data_feed": data_feed,
                "streams": self.streams,
            },
        )
    
    @property
    def is_connected(self) -> bool:
        return self._connected
    
    @property
    def state(self) -> str:
        return self._state
    
    @property
    def order_book(self):
        """Alpaca no proporciona order book L2 en paper - retorna None para compatibilidad."""
        return None
    
    @property
    def order_book_metrics(self):
        return None
    
    @property
    def best_bid(self) -> Optional[float]:
        return None
    
    @property
    def best_ask(self) -> Optional[float]:
        return None
    
    @property
    def mid_price(self) -> Optional[float]:
        return None
    
    @property
    def spread(self) -> Optional[float]:
        return None
    
    def _set_state(self, new_state: str) -> None:
        old = self._state
        self._state = new_state
        if self.on_state_change and old != new_state:
            try:
                self.on_state_change(old, new_state)
            except Exception as e:
                logger.error("Error in state change callback", extra={"error": str(e)}, exc_info=True)
    
    async def connect(self) -> None:
        """Inicia conexión WebSocket."""
        if self._running:
            logger.warning("Feed already running")
            return
        
        self._running = True
        self._set_state(ConnectionState.CONNECTING)
        
        try:
            await self._connect_websocket()
            self._connected = True
            self._set_state(ConnectionState.CONNECTED)
            
            # Enviar suscripción
            await self._send_subscription()
            
            # Iniciar tasks
            self._emit_task = asyncio.create_task(self._emit_loop())
            self._health_task = asyncio.create_task(self._health_loop())
            self._ping_task = asyncio.create_task(self._ping_loop())
            
            logger.info("AlpacaWSFeed connected", extra={"symbol": self.symbol})
            
        except Exception as e:
            self._connected = False
            self._set_state(ConnectionState.ERROR)
            logger.error("AlpacaWSFeed connection failed", extra={"symbol": self.symbol, "error": str(e)})
            raise
    
    async def _connect_websocket(self) -> None:
        """Conecta WebSocket con autenticación."""
        import websockets
        
        # Alpaca requiere autenticación al conectar
        self._ws = await websockets.connect(
            self.ws_url,
            ping_interval=None,  # Manejamos ping manualmente
            ping_timeout=None,
            close_timeout=10,
        )
        
        # Enviar autenticación
        auth_msg = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.api_secret,
        }
        await self._ws.send(json.dumps(auth_msg))
        
        # Esperar confirmación de conexión (primero viene "connected")
        conn_response = await self._ws.recv()
        conn_data = json.loads(conn_response)
        if isinstance(conn_data, list) and len(conn_data) > 0:
            if conn_data[0].get("T") == "success" and conn_data[0].get("msg") == "connected":
                logger.debug("Alpaca WebSocket connected", extra={"symbol": self.symbol})
            else:
                raise RuntimeError(f"Connection failed: {conn_data}")
        else:
            raise RuntimeError(f"Unexpected connection response: {conn_data}")
        
        # Esperar confirmación de autenticación
        auth_response = await self._ws.recv()
        auth_data = json.loads(auth_response)
        if isinstance(auth_data, list) and len(auth_data) > 0:
            if auth_data[0].get("T") == "success" and auth_data[0].get("msg") == "authenticated":
                logger.info("Alpaca WebSocket authenticated", extra={"symbol": self.symbol})
            else:
                raise RuntimeError(f"Authentication failed: {auth_data}")
        else:
            raise RuntimeError(f"Unexpected auth response: {auth_data}")
    
    async def _send_subscription(self) -> None:
        """Envía suscripción a streams."""
        subscribe_msg = {
            "action": "subscribe",
            self.streams[0]: self.streams[1:] if len(self.streams) > 1 else self.streams[0]
        } if len(self.streams) == 1 else {
            "action": "subscribe",
            "trades": [self.symbol] if "trades" in self.streams else [],
            "quotes": [self.symbol] if "quotes" in self.streams else [],
            "bars": [self.symbol] if "bars" in self.streams else [],
        }
        # Simplificado: suscribir a todos los streams para el símbolo
        subscribe_msg = {
            "action": "subscribe",
            "trades": [self.symbol] if "trades" in self.streams else [],
            "quotes": [self.symbol] if "quotes" in self.streams else [],
            "bars": [self.symbol] if "bars" in self.streams else [],
        }
        # Filtrar streams vacíos
        subscribe_msg = {k: v for k, v in subscribe_msg.items() if v}
        
        await self._ws.send(json.dumps(subscribe_msg))
        logger.info("Alpaca subscription sent", extra={"symbol": self.symbol, "streams": self.streams})
    
    async def disconnect(self) -> None:
        """Desconecta limpiamente."""
        self._running = False
        
        for task in [self._emit_task, self._health_task, self._ping_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
        
        self._connected = False
        self._set_state(ConnectionState.DISCONNECTED)
        logger.info("AlpacaWSFeed disconnected", extra={"symbol": self.symbol})
    
    async def _emit_loop(self) -> None:
        """Loop que procesa mensajes WS y emite observaciones."""
        try:
            while self._running and self._ws:
                try:
                    message = await asyncio.wait_for(self._ws.recv(), timeout=1.0)
                    
                    self.stats.events_received += 1
                    receive_time = time.time()
                    
                    # Procesar mensaje
                    observations = self._process_raw_message(message, receive_time)
                    
                    for obs in observations:
                        self.stats.events_emitted += 1
                        self.stats.last_emitted_time = time.time()
                        
                        # Calcular latencia
                        latency_ms = (time.time() - obs.timestamp) * 1000
                        self.stats.avg_latency_ms = (
                            self.stats.avg_latency_ms * 0.99 + latency_ms * 0.01
                        )
                        self.stats.max_latency_ms = max(self.stats.max_latency_ms, latency_ms)
                        
                        # Enqueue
                        try:
                            self._obs_queue.put_nowait(obs)
                        except asyncio.QueueFull:
                            logger.warning("Observation queue full, dropping", extra={"symbol": self.symbol})
                        
                        # Callback
                        if self.on_observation:
                            try:
                                self.on_observation(obs)
                            except Exception as e:
                                logger.error("Error in on_observation callback", extra={"error": str(e)})
                                
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error("Emit loop error", extra={"symbol": self.symbol, "error": str(e)})
                    if self._running:
                        await self._reconnect()
                        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error("Emit loop fatal error", extra={"symbol": self.symbol, "error": str(e)})
    
    async def _reconnect(self) -> None:
        """Reconexión automática con backoff."""
        self._connected = False
        self._set_state(ConnectionState.RECONNECTING)
        self._reconnect_count += 1
        
        base_delay = 1.0
        max_delay = 60.0
        delay = min(base_delay * (2 ** min(self._reconnect_count, 6)), max_delay)
        
        logger.info(f"Reconnecting in {delay:.1f}s...", extra={"symbol": self.symbol, "attempt": self._reconnect_count})
        await asyncio.sleep(delay)
        
        if self._running:
            try:
                await self._connect_websocket()
                await self._send_subscription()
                self._connected = True
                self._set_state(ConnectionState.CONNECTED)
                self.stats.reconnects += 1
                logger.info("Reconnected successfully", extra={"symbol": self.symbol})
            except Exception as e:
                logger.error("Reconnection failed", extra={"symbol": self.symbol, "error": str(e)})
                await self._reconnect()
    
    async def _ping_loop(self) -> None:
        """Envía ping periódico para mantener conexión viva."""
        while self._running:
            await asyncio.sleep(20)
            if self._ws and self._connected:
                try:
                    await self._ws.ping()
                    self._last_ping = time.time()
                except Exception as e:
                    logger.warning("Ping failed, will reconnect", extra={"error": str(e)})
                    await self._reconnect()
    
    def _process_raw_message(self, message: str, receive_time: float) -> List[MarketObservation]:
        """Convierte mensaje raw WS a lista de MarketObservation."""
        observations = []
        
        try:
            data = json.loads(message)
            
            # Alpaca envía array de mensajes
            if isinstance(data, list):
                for msg in data:
                    observations.extend(self._process_single_message(msg, receive_time))
            else:
                observations.extend(self._process_single_message(data, receive_time))
                
        except json.JSONDecodeError as e:
            logger.error("JSON decode error", extra={"error": str(e)})
        except Exception as e:
            logger.error("Error processing message", extra={"error": str(e)})
        
        return observations
    
    def _process_single_message(self, msg: Dict, receive_time: float) -> List[MarketObservation]:
        """Procesa un mensaje individual."""
        observations = []
        msg_type = msg.get("T", "")
        
        try:
            if msg_type == "t":  # Trade
                self.stats.trades_received += 1
                obs = self._create_trade(msg, receive_time)
                observations.append(obs)
                
            elif msg_type == "q":  # Quote
                self.stats.quotes_received += 1
                try:
                    obs = self._create_quote(msg, receive_time)
                except ValueError as e:
                    # Validación del dominio (p. ej. ask=0 con mercado
                    # cerrado / feed IEX gratuito): descartar en silencio.
                    self.stats.quotes_stale_dropped += 1
                    logger.debug("Stale quote dropped (validación)", extra={"error": str(e)})
                    obs = None
                if obs is None:
                    pass  # Ya contabilizada como stale arriba.
                elif (
                    obs.bid <= 0 or obs.ask <= 0 or obs.ask < obs.bid
                ):
                    # Cotización stale (típico con mercado cerrado / feed IEX
                    # gratuito): no emitir para no contaminar el engine.
                    self.stats.quotes_stale_dropped += 1
                    logger.debug(
                        "Stale quote dropped",
                        extra={"symbol": obs.symbol, "bid": obs.bid, "ask": obs.ask},
                    )
                elif obs is not None:
                    observations.append(obs)
                
            elif msg_type == "b":  # Bar
                self.stats.bars_received += 1
                obs = self._create_candle(msg, receive_time)
                observations.append(obs)
                
            elif msg_type == "success":
                logger.debug("Alpaca success message", extra={"msg": msg.get("msg")})
                
            elif msg_type == "error":
                logger.error("Alpaca error message", extra={"msg": msg.get("msg"), "code": msg.get("code")})
                
            elif msg_type == "subscription":
                logger.info("Alpaca subscription confirmed", extra={"streams": msg.get("streams")})
                
        except Exception as e:
            logger.error("Error processing single message", extra={"error": str(e), "type": msg_type})
        
        return observations
    
    def _create_trade(self, data: Dict, receive_time: float) -> Trade:
        """Crea Trade desde mensaje Alpaca."""
        from iot_machine_learning.infrastructure.adapters.market.alpaca.helpers import iso_to_epoch, as_float, require
        
        return Trade(
            symbol=str(data.get("S", self.symbol)),
            timestamp=iso_to_epoch(str(require(data, "t"))),
            data_status=DataStatus.REALTIME,
            source_provider="alpaca",
            price=as_float(require(data, "p"), "p"),
            size=as_float(require(data, "s"), "s"),
            trade_id=str(require(data, "i")),
            taker_side=None,  # Alpaca no indica taker side en trades
            conditions=tuple(str(c) for c in require(data, "c", label="conditions")),
            tape=str(require(data, "z", label="tape")),
            corrected=bool(require(data, "u", label="corrected")),
        )
    
    def _create_quote(self, data: Dict, receive_time: float) -> Quote:
        """Crea Quote desde mensaje Alpaca."""
        from iot_machine_learning.infrastructure.adapters.market.alpaca.helpers import iso_to_epoch, as_float, require
        
        return Quote(
            symbol=str(data.get("S", self.symbol)),
            timestamp=iso_to_epoch(str(require(data, "t"))),
            data_status=DataStatus.REALTIME,
            source_provider="alpaca",
            venue=str(require(data, "bx", label="bid exchange")),
            bid=as_float(require(data, "bp"), "bp"),
            bid_size=as_float(require(data, "bs"), "bs"),
            ask=as_float(require(data, "ap"), "ap"),
            ask_size=as_float(require(data, "as"), "as"),
            bid_exchange=str(require(data, "bx", label="bid exchange")),
            ask_exchange=str(require(data, "ax", label="ask exchange")),
            conditions=tuple(str(c) for c in require(data, "c", label="conditions")),
            tape=str(require(data, "z", label="tape")),
        )
    
    def _create_candle(self, data: Dict, receive_time: float) -> Candle:
        """Crea Candle desde mensaje Alpaca."""
        from iot_machine_learning.infrastructure.adapters.market.alpaca.helpers import iso_to_epoch, as_float, require
        
        # Convertir intervalo a segundos
        interval_map = {
            "1Min": 60, "5Min": 300, "15Min": 900, "30Min": 1800,
            "1Hour": 3600, "1Day": 86400
        }
        interval_seconds = interval_map.get(self.bar_interval, 60)
        
        return Candle(
            symbol=str(data.get("S", self.symbol)),
            timestamp=iso_to_epoch(str(require(data, "t"))),
            data_status=DataStatus.REALTIME,
            source_provider="alpaca",
            venue=None,
            open=as_float(require(data, "o"), "o"),
            high=as_float(require(data, "h"), "h"),
            low=as_float(require(data, "l"), "l"),
            close=as_float(require(data, "c"), "c"),
            volume=as_float(require(data, "v"), "v"),
            interval_seconds=interval_seconds,
            vwap=as_float(data.get("vw", 0), "vw") if data.get("vw") else None,
            trade_count=int(require(data, "n", label="trade_count")),
            adjusted=False,
        )
    
    async def _health_loop(self) -> None:
        """Loop de salud: métricas periódicas."""
        while self._running:
            await asyncio.sleep(10)
            
            if not self._running:
                break
            
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
        """Obtiene próxima observación."""
        try:
            if timeout is None:
                return await self._obs_queue.get()
            return await asyncio.wait_for(self._obs_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    def get_stats(self) -> Dict:
        """Estadísticas completas del feed."""
        return {
            "symbol": self.symbol,
            "running": self._running,
            "connected": self._connected,
            "state": self._state,
            "feed_stats": self.stats.to_dict(),
            "reconnect_count": self._reconnect_count,
        }


# Alias para compatibilidad
AlpacaLiveFeed = AlpacaWSFeed