"""Binance WebSocket Client — conexiones asyncio con reconexión exponencial.

Implementa:
- Conexión persistente a streams Binance (depth, aggTrade, bookTicker)
- Reconexión automática con backoff exponencial + jitter
- Ping/Pong keepalive + detección de conexión muerta
- Buffer de eventos entrantes con backpressure
- Métricas de latencia y estado de conexión
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Deque, Dict, List, Optional, Set
from urllib.parse import urlencode

import aiohttp
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """Estados de la conexión WebSocket."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    DEGRADED = "degraded"
    SHUTTING_DOWN = "shutting_down"


@dataclass
class ConnectionMetrics:
    """Métricas de la conexión para monitoreo."""
    state: ConnectionState = ConnectionState.DISCONNECTED
    connected_at: Optional[float] = None
    last_message_at: Optional[float] = None
    last_pong_at: Optional[float] = None
    messages_received: int = 0
    messages_sent: int = 0
    reconnect_count: int = 0
    total_downtime: float = 0.0
    last_error: Optional[str] = None
    ping_latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "connected_at": self.connected_at,
            "last_message_at": self.last_message_at,
            "messages_received": self.messages_received,
            "reconnect_count": self.reconnect_count,
            "total_downtime": self.total_downtime,
            "ping_latency_ms": self.ping_latency_ms,
            "last_error": self.last_error,
        }


class BinanceWSClient:
    """
    Cliente WebSocket para Binance con reconexión automática.
    
    Soporta streams combinados: depth@100ms, aggTrade, bookTicker, kline.
    Maneja buffer de mensajes con backpressure y métricas de salud.
    """
    
    # Configuración de reconexión
    BASE_RECONNECT_DELAY = 1.0      # segundos
    MAX_RECONNECT_DELAY = 60.0      # segundos
    RECONNECT_JITTER = 0.3          # 30% jitter
    PING_INTERVAL = 20.0            # segundos
    PING_TIMEOUT = 10.0             # segundos
    MAX_QUEUE_SIZE = 10000          # backpressure limit
    MESSAGE_TIMEOUT = 30.0          # timeout para recibir mensaje
    
    BASE_URL = "wss://stream.binance.com:9443/ws"
    TESTNET_BASE_URL = "wss://stream.binancefuture.com/ws"  # USD-M Futures testnet
    
    def __init__(
        self,
        symbol: str,
        streams: List[str],
        *,
        testnet: bool = False,
        on_message: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_state_change: Optional[Callable[[ConnectionState, ConnectionState], None]] = None,
        max_queue_size: int = MAX_QUEUE_SIZE,
    ):
        """
        Args:
            symbol: Símbolo normalizado (ej. "BTCUSDT")
            streams: Lista de streams (ej. ["btcusdt@depth@100ms", "btcusdt@aggTrade"])
            testnet: Usar Binance Futures Testnet
            on_message: Callback opcional para cada mensaje parseado
            on_state_change: Callback opcional para cambios de estado
            max_queue_size: Tamaño máximo de la cola interna
        """
        self.symbol = symbol.upper()
        self.streams = streams
        self.testnet = testnet
        self.on_message = on_message
        self.on_state_change = on_state_change
        
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=max_queue_size)
        self._state = ConnectionState.DISCONNECTED
        self._metrics = ConnectionMetrics()
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._tasks: Set[asyncio.Task] = set()
        self._shutdown = False
        self._reconnect_lock = asyncio.Lock()
        self._last_update_id: Optional[int] = None  # Para sincronización order book
        
        # URL base
        self.base_url = self.TESTNET_BASE_URL if testnet else self.BASE_URL
        
        logger.info(
            "BinanceWSClient initialized",
            extra={
                "symbol": self.symbol,
                "streams": self.streams,
                "testnet": testnet,
            },
        )
    
    @property
    def state(self) -> ConnectionState:
        return self._state
    
    @property
    def metrics(self) -> ConnectionMetrics:
        return self._metrics
    
    @property
    def is_connected(self) -> bool:
        return self._state == ConnectionState.CONNECTED
    
    @property
    def queue_size(self) -> int:
        return self._queue.qsize()
    
    def _set_state(self, new_state: ConnectionState) -> None:
        """Cambia estado y notifica callback."""
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            self._metrics.state = new_state
            if new_state == ConnectionState.CONNECTED:
                self._metrics.connected_at = time.time()
            logger.info(
                "WebSocket state change",
                extra={"from": old_state.value, "to": new_state.value, "symbol": self.symbol},
            )
            if self.on_state_change:
                try:
                    self.on_state_change(old_state, new_state)
                except Exception as e:
                    logger.error("Error in on_state_change callback", extra={"error": str(e)})
    
    async def connect(self) -> None:
        """Inicia conexión y tareas de fondo."""
        if self._shutdown:
            raise RuntimeError("Client is shut down")
        if self._state in (ConnectionState.CONNECTED, ConnectionState.CONNECTING):
            logger.warning("Already connected or connecting")
            return
        
        self._shutdown = False
        self._set_state(ConnectionState.CONNECTING)
        
        # Crear sesión HTTP para REST snapshots
        timeout = aiohttp.ClientTimeout(total=10, connect=5)
        self._session = aiohttp.ClientSession(timeout=timeout)
        
        # Iniciar tarea principal de conexión
        task = asyncio.create_task(self._run_forever())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        
        # Esperar a que conecte o falle
        for _ in range(50):  # 5 segundos max
            if self._state == ConnectionState.CONNECTED:
                logger.info("WebSocket connected", extra={"symbol": self.symbol})
                return
            if self._state == ConnectionState.DISCONNECTED and self._shutdown:
                raise RuntimeError("Connection failed during startup")
            await asyncio.sleep(0.1)
        
        raise TimeoutError("WebSocket connection timeout")
    
    async def disconnect(self) -> None:
        """Cierra conexión limpiamente."""
        self._shutdown = True
        self._set_state(ConnectionState.SHUTTING_DOWN)
        
        # Cancelar tareas
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        # Cerrar WebSocket
        if self._ws and not self._ws.closed:
            await self._ws.close()
        
        # Cerrar sesión HTTP
        if self._session and not self._session.closed:
            await self._session.close()
        
        self._set_state(ConnectionState.DISCONNECTED)
        logger.info("WebSocket disconnected", extra={"symbol": self.symbol})
    
    async def _run_forever(self) -> None:
        """Loop principal: conecta, escucha, reconecta si falla."""
        consecutive_failures = 0
        
        while not self._shutdown:
            try:
                await self._connect_and_listen()
                consecutive_failures = 0  # Reset en éxito
            except asyncio.CancelledError:
                raise
            except Exception as e:
                consecutive_failures += 1
                self._metrics.last_error = str(e)
                logger.error(
                    "WebSocket error, will reconnect",
                    extra={
                        "symbol": self.symbol,
                        "error": str(e),
                        "consecutive_failures": consecutive_failures,
                    },
                )
                
                if self._shutdown:
                    break
                
                # Backoff exponencial con jitter
                delay = min(
                    self.BASE_RECONNECT_DELAY * (2 ** (consecutive_failures - 1)),
                    self.MAX_RECONNECT_DELAY,
                )
                delay *= (1 + random.uniform(-self.RECONNECT_JITTER, self.RECONNECT_JITTER))
                
                self._set_state(ConnectionState.RECONNECTING)
                self._metrics.reconnect_count += 1
                downtime_start = time.time()
                
                logger.info(
                    "Reconnecting",
                    extra={"symbol": self.symbol, "delay": delay, "attempt": consecutive_failures},
                )
                
                await asyncio.sleep(delay)
                
                self._metrics.total_downtime += time.time() - downtime_start
    
    async def _connect_and_listen(self) -> None:
        """Conecta y procesa mensajes hasta que falle."""
        stream_path = "/".join(self.streams)
        url = f"{self.base_url}/{stream_path}"
        
        logger.debug("Connecting to WebSocket", extra={"url": url, "symbol": self.symbol})
        
        # Headers para Binance
        extra_headers = {
            "User-Agent": "ZENIN-Bot/1.0",
        }
        
        async with websockets.connect(
            url,
            extra_headers=extra_headers,
            ping_interval=None,  # Manejamos ping manualmente
            ping_timeout=self.PING_TIMEOUT,
            close_timeout=5,
            max_size=2**20,  # 1MB max message
            compression=None,  # Deshabilitar compresión para latencia
        ) as ws:
            self._ws = ws
            self._set_state(ConnectionState.CONNECTED)
            self._metrics.last_message_at = time.time()
            
            # Iniciar tareas de ping y procesamiento
            ping_task = asyncio.create_task(self._ping_loop())
            process_task = asyncio.create_task(self._process_messages())
            
            try:
                await asyncio.gather(ping_task, process_task)
            except asyncio.CancelledError:
                ping_task.cancel()
                process_task.cancel()
                raise
            finally:
                ping_task.cancel()
                process_task.cancel()
                await asyncio.gather(ping_task, process_task, return_exceptions=True)
    
    async def _ping_loop(self) -> None:
        """Envía ping periódico y verifica pong."""
        while not self._shutdown and self._state == ConnectionState.CONNECTED:
            await asyncio.sleep(self.PING_INTERVAL)
            
            if self._shutdown or self._state != ConnectionState.CONNECTED:
                break
            
            try:
                ping_start = time.perf_counter()
                # Binance usa ping/pong nativo de WebSocket
                pong_waiter = self._ws.ping()
                await asyncio.wait_for(pong_waiter, timeout=self.PING_TIMEOUT)
                self._metrics.ping_latency_ms = (time.perf_counter() - ping_start) * 1000
                self._metrics.last_pong_at = time.time()
                
                # Detectar latencia alta
                if self._metrics.ping_latency_ms > 1000:
                    logger.warning(
                        "High ping latency",
                        extra={"symbol": self.symbol, "latency_ms": self._metrics.ping_latency_ms},
                    )
                    self._set_state(ConnectionState.DEGRADED)
                elif self._state == ConnectionState.DEGRADED:
                    self._set_state(ConnectionState.CONNECTED)
                    
            except asyncio.TimeoutError:
                logger.warning("Ping timeout, connection may be dead", extra={"symbol": self.symbol})
                self._set_state(ConnectionState.DEGRADED)
                # Forzar cierre para trigger reconexión
                await self._ws.close()
                break
            except Exception as e:
                logger.error("Ping error", extra={"symbol": self.symbol, "error": str(e)})
                break
    
    async def _process_messages(self) -> None:
        """Procesa mensajes entrantes del WebSocket."""
        try:
            async for raw_message in self._ws:
                if self._shutdown:
                    break
                
                receive_time = time.time()
                self._metrics.last_message_at = receive_time
                self._metrics.messages_received += 1
                
                try:
                    message = json.loads(raw_message)
                except json.JSONDecodeError as e:
                    logger.warning("Invalid JSON from WebSocket", extra={"error": str(e)})
                    continue
                
                # Manejar pong (respuesta a nuestro ping)
                if isinstance(message, dict) and message.get("result") is None and "id" not in message:
                    # Podría ser pong frame - websockets lo maneja internamente
                    pass
                
                # Añadir timestamp de recepción
                message["_received_at"] = receive_time
                
                # Enqueue con backpressure
                try:
                    self._queue.put_nowait(message)
                except asyncio.QueueFull:
                    logger.warning(
                        "Message queue full, dropping message",
                        extra={"symbol": self.symbol, "queue_size": self._queue.qsize()},
                    )
                    # Drop oldest si es crítico, o simplemente skip
                    try:
                        self._queue.get_nowait()
                        self._queue.put_nowait(message)
                    except asyncio.QueueEmpty:
                        pass
                
                # Callback opcional
                if self.on_message:
                    try:
                        self.on_message(message)
                    except Exception as e:
                        logger.error("Error in on_message callback", extra={"error": str(e)})
                        
        except ConnectionClosed as e:
            logger.info(
                "WebSocket connection closed",
                extra={"symbol": self.symbol, "code": e.code, "reason": e.reason},
            )
            raise
        except WebSocketException as e:
            logger.error("WebSocket error", extra={"symbol": self.symbol, "error": str(e)})
            raise
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("Unexpected error in message processing", extra={"symbol": self.symbol, "error": str(e)})
            raise
    
    async def get_message(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """
        Obtiene el siguiente mensaje de la cola.
        
        Args:
            timeout: Segundos a esperar (None = indefinido)
            
        Returns:
            Mensaje parseado o None si timeout/shutdown
        """
        if timeout is None:
            return await self._queue.get()
        try:
            return await asyncio.wait_for(self._queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
    
    async def iter_messages(self) -> AsyncGenerator[Dict[str, Any], None]:
        """Iterador asíncrono sobre mensajes."""
        while not self._shutdown:
            msg = await self.get_message()
            if msg is None:
                if self._shutdown:
                    break
                continue
            yield msg
    
    async def get_snapshot_rest(self) -> Dict[str, Any]:
        """
        Obtiene snapshot REST del order book para sincronización inicial.
        
        Returns:
            Dict con lastUpdateId, bids, asks
        """
        if not self._session:
            raise RuntimeError("HTTP session not initialized")
        
        symbol = self.symbol
        url = f"https://{'testnet.' if self.testnet else ''}binance.com/api/v3/depth"
        params = {"symbol": symbol, "limit": 1000}
        
        async with self._session.get(url, params=params) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data
    
    def get_connection_info(self) -> Dict[str, Any]:
        """Información completa de la conexión para health checks."""
        return {
            "symbol": self.symbol,
            "streams": self.streams,
            "testnet": self.testnet,
            "state": self._state.value,
            "metrics": self._metrics.to_dict(),
            "queue_size": self._queue.qsize(),
            "tasks_running": len([t for t in self._tasks if not t.done()]),
        }


# Factory para streams comunes
def create_market_streams(symbol: str, depth_speed: str = "100ms") -> List[str]:
    """
    Crea lista de streams estándar para trading.
    
    Args:
        symbol: Símbolo en minúsculas (ej. "btcusdt")
        depth_speed: "100ms" o "1000ms"
        
    Returns:
        Lista de streams para Binance
    """
    sym = symbol.lower()
    return [
        f"{sym}@depth@{depth_speed}",      # Order book L2 updates
        f"{sym}@aggTrade",                  # Aggregated trades
        f"{sym}@bookTicker",                # Best bid/ask
        # f"{sym}@kline_1m",                # Opcional: velas 1m
    ]


# Context manager para uso sencillo
@asynccontextmanager
async def binance_ws_connection(
    symbol: str,
    streams: Optional[List[str]] = None,
    testnet: bool = False,
    **kwargs,
) -> AsyncGenerator[BinanceWSClient, None]:
    """
    Context manager para conexión WebSocket limpia.
    
    Uso:
        async with binance_ws_connection("BTCUSDT") as client:
            async for msg in client.iter_messages():
                process(msg)
    """
    if streams is None:
        streams = create_market_streams(symbol)
    
    client = BinanceWSClient(symbol, streams, testnet=testnet, **kwargs)
    try:
        await client.connect()
        yield client
    finally:
        await client.disconnect()


