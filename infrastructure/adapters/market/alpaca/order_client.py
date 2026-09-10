"""AlpacaOrderClient -- Cliente REST firmado para Alpaca Paper Trading.

Implementa BrokerClientProtocol para el execution handler.
Rate limiting, retry con backoff, order types: MARKET, LIMIT, STOP, STOP_LIMIT, TRAILING_STOP.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from collections import deque

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """Solicitud de orden estandarizada."""
    symbol: str
    side: str              # "buy" o "sell"
    order_type: str        # "market", "limit", "stop", "stop_limit", "trailing_stop"
    qty: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    trail_price: Optional[float] = None
    trail_percent: Optional[float] = None
    time_in_force: str = "day"  # day, gtc, opg, cls, ioc, fok
    extended_hours: bool = False
    client_order_id: Optional[str] = None
    order_class: Optional[str] = None  # simple, bracket, oco, oto
    take_profit: Optional[Dict] = None
    stop_loss: Optional[Dict] = None


@dataclass
class OrderResponse:
    """Respuesta de orden de Alpaca."""
    id: str
    client_order_id: str
    symbol: str
    status: str           # new, partially_filled, filled, done_for_day, canceled, expired, replaced, pending_cancel, pending_replace, accepted, pending_new, accepted_for_bidding, stopped, suspended, calculated
    side: str
    order_type: str
    qty: float
    price: Optional[float]
    stop_price: Optional[float]
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    commission: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0
    raw: Dict = field(default_factory=dict)


class RateLimiter:
    """Rate limiter token bucket para Alpaca API."""

    def __init__(
        self,
        max_requests_per_minute: int = 200,
        max_orders_per_second: int = 10,
    ):
        self.max_rpm = max_requests_per_minute
        self.max_ops = max_orders_per_second

        self._request_times: deque = deque(maxlen=max_requests_per_minute)
        self._order_times: deque = deque(maxlen=max_orders_per_second)

        self._lock = asyncio.Lock()

    async def acquire(self, weight: int = 1) -> None:
        """Espera hasta que haya capacidad."""
        async with self._lock:
            now = time.time()

            # Rate limit per minute
            cutoff_min = now - 60
            while self._request_times and self._request_times[0] < cutoff_min:
                self._request_times.popleft()

            if len(self._request_times) >= self.max_rpm:
                wait = 60 - (now - self._request_times[0])
                await asyncio.sleep(max(0, wait) + 0.1)

            # Rate limit per second (orders)
            cutoff_sec = now - 1
            while self._order_times and self._order_times[0] < cutoff_sec:
                self._order_times.popleft()

            if len(self._order_times) >= self.max_ops:
                wait = 1 - (now - self._order_times[0])
                await asyncio.sleep(max(0, wait) + 0.01)

            self._request_times.append(now)
            self._order_times.append(now)


class AlpacaOrderClient:
    """
    Cliente REST firmado para Alpaca Paper Trading (Equities).

    Características:
    - Autenticación via API Key / Secret (header-based)
    - Rate limiting automático
    - Retry con backoff exponencial
    - Tipos de orden: MARKET, LIMIT, STOP, STOP_LIMIT, TRAILING_STOP
    - Cancelación individual y masiva
    - Consulta de posiciones y órdenes abiertas
    """

    BASE_URL = "https://paper-api.alpaca.markets/v2"
    LIVE_BASE_URL = "https://api.alpaca.markets/v2"

    # Order types (Alpaca)
    ORDER_TYPE_MARKET = "market"
    ORDER_TYPE_LIMIT = "limit"
    ORDER_TYPE_STOP = "stop"
    ORDER_TYPE_STOP_LIMIT = "stop_limit"
    ORDER_TYPE_TRAILING_STOP = "trailing_stop"

    # Time in force
    TIME_IN_FORCE_DAY = "day"
    TIME_IN_FORCE_GTC = "gtc"
    TIME_IN_FORCE_OPG = "opg"
    TIME_IN_FORCE_CLS = "cls"
    TIME_IN_FORCE_IOC = "ioc"
    TIME_IN_FORCE_FOK = "fok"

    # Order statuses
    STATUS_NEW = "new"
    STATUS_PARTIALLY_FILLED = "partially_filled"
    STATUS_FILLED = "filled"
    STATUS_DONE_FOR_DAY = "done_for_day"
    STATUS_CANCELED = "canceled"
    STATUS_EXPIRED = "expired"
    STATUS_REPLACED = "replaced"
    STATUS_PENDING_CANCEL = "pending_cancel"
    STATUS_PENDING_REPLACE = "pending_replace"
    STATUS_ACCEPTED = "accepted"
    STATUS_PENDING_NEW = "pending_new"
    STATUS_ACCEPTED_FOR_BIDDING = "accepted_for_bidding"
    STATUS_STOPPED = "stopped"
    STATUS_SUSPENDED = "suspended"
    STATUS_CALCULATED = "calculated"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: Optional[str] = None,
        *,
        data_feed: str = "iex",
        recv_window: int = 5000,
        max_retries: int = 3,
        base_retry_delay: float = 0.5,
        max_retry_delay: float = 10.0,
    ):
        """
        Args:
            api_key: API Key de Alpaca
            api_secret: API Secret de Alpaca
            base_url: URL base personalizada para trading API (opcional, default paper)
            data_feed: "iex" o "sip" para market data
            recv_window: Ventana de recepción en ms (no usado en Alpaca, mantenido para compat)
            max_retries: Reintentos máximos para requests fallidos
            base_retry_delay: Delay base para backoff exponencial
            max_retry_delay: Delay máximo para reintentos
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url or self.BASE_URL
        self.data_feed_base_url = "https://data.alpaca.markets/v2"
        self.data_feed = data_feed
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.max_retry_delay = max_retry_delay

        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = RateLimiter()
        self._closed = False

        # Métricas
        self._request_count = 0
        self._error_count = 0
        self._latencies: deque = deque(maxlen=1000)

        logger.info(
            "AlpacaOrderClient initialized",
            extra={"base_url": self.base_url, "data_feed": self.data_feed},
        )

    async def __aenter__(self) -> "AlpacaOrderClient":
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _ensure_session(self) -> None:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=20)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={
                    "APCA-API-KEY-ID": self.api_key,
                    "APCA-API-SECRET-KEY": self.api_secret,
                },
            )

    async def close(self) -> None:
        """Cierra la sesión HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._closed = True

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        weight: int = 1,
        use_data_api: bool = False,
    ) -> Dict:
        """Ejecuta request HTTP con rate limiting, retry y métricas."""
        await self._rate_limiter.acquire(weight)

        await self._ensure_session()
        base = self.data_feed_base_url if use_data_api else self.base_url
        url = f"{base}{endpoint}"

        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

        if method in ("POST", "PUT", "PATCH"):
            headers["Content-Type"] = "application/json"

        last_exception = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                async with self._session.request(
                    method, url, params=params, json=json_data, headers=headers
                ) as resp:
                    latency = time.perf_counter() - start
                    self._latencies.append(latency)

                    if resp.status in (200, 201):
                        self._request_count += 1
                        return await resp.json()
                    elif resp.status == 204:
                        # No Content - éxito sin body (ej. DELETE order)
                        self._request_count += 1
                        return {"status": "success", "code": 204}
                    elif resp.status == 429:
                        # Rate limited - esperar y reintentar
                        retry_after = int(resp.headers.get("Retry-After", 1))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    elif resp.status == 401:
                        # Unauthorized
                        error_text = await resp.text()
                        raise RuntimeError(f"Authentication failed: {error_text}")
                    elif resp.status == 403:
                        # Forbidden
                        error_text = await resp.text()
                        raise RuntimeError(f"Forbidden: {error_text}")
                    else:
                        error_text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {error_text}")

            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError("Request timeout")
            except aiohttp.ClientError as e:
                last_exception = e

            # Backoff exponencial
            if attempt < self.max_retries:
                delay = min(
                    self.base_retry_delay * (2 ** attempt),
                    self.max_retry_delay,
                )
                await asyncio.sleep(delay)

        self._error_count += 1
        raise RuntimeError(f"Request failed after {self.max_retries + 1} attempts: {last_exception}")

    # --- Public API ---

    async def get_account(self) -> Dict:
        """Obtiene información de la cuenta."""
        return await self._request("GET", "/v2/account", weight=1)

    async def get_account_configurations(self) -> Dict:
        """Obtiene configuraciones de la cuenta."""
        return await self._request("GET", "/v2/account/configurations", weight=1)

    # --- Positions ---

    async def get_positions(self) -> List[Dict]:
        """Obtiene todas las posiciones abiertas."""
        return await self._request("GET", "/v2/positions", weight=1)

    async def get_position(self, symbol: str) -> Dict:
        """Obtiene posición para un símbolo específico. Retorna qty=0 si no existe."""
        try:
            return await self._request("GET", f"/v2/positions/{symbol.upper()}", weight=1)
        except RuntimeError as e:
            if "404" in str(e) or "position does not exist" in str(e):
                # No position - return empty position dict
                return {"symbol": symbol.upper(), "qty": "0", "side": "long", "avg_entry_price": "0", "market_value": "0"}
            raise

    async def close_position(self, symbol: str) -> Dict:
        """Cierra una posición específica a mercado."""
        try:
            return await self._request("DELETE", f"/v2/positions/{symbol.upper()}", weight=1)
        except RuntimeError as e:
            if "404" in str(e) or "position does not exist" in str(e):
                # No position to close
                return {"symbol": symbol.upper(), "status": "closed", "qty": "0"}
            raise

    async def close_all_positions(self, cancel_orders: bool = True) -> List[Dict]:
        """Cierra todas las posiciones."""
        params = {"cancel_orders": "true" if cancel_orders else "false"}
        return await self._request("DELETE", "/v2/positions", params=params, weight=1)

    # --- Orders ---

    async def submit_order(self, request: OrderRequest) -> OrderResponse:
        """Envía orden a Alpaca."""
        params = {
            "symbol": request.symbol.upper(),
            "side": request.side.lower(),
            "type": request.order_type,
            "qty": str(request.qty),
            "time_in_force": request.time_in_force,
        }

        if request.order_type in (self.ORDER_TYPE_LIMIT, self.ORDER_TYPE_STOP_LIMIT):
            if request.price is None:
                raise ValueError(f"{request.order_type} requires price")
            params["limit_price"] = str(request.price)

        if request.order_type in (self.ORDER_TYPE_STOP, self.ORDER_TYPE_STOP_LIMIT):
            if request.stop_price is None:
                raise ValueError(f"{request.order_type} requires stop_price")
            params["stop_price"] = str(request.stop_price)

        if request.order_type == self.ORDER_TYPE_TRAILING_STOP:
            if request.trail_price is None and request.trail_percent is None:
                raise ValueError("trailing_stop requires trail_price or trail_percent")
            if request.trail_price is not None:
                params["trail_price"] = str(request.trail_price)
            if request.trail_percent is not None:
                params["trail_percent"] = str(request.trail_percent)

        if request.extended_hours:
            params["extended_hours"] = "true"

        if request.client_order_id:
            params["client_order_id"] = request.client_order_id

        if request.order_class:
            params["order_class"] = request.order_class
            if request.take_profit:
                params["take_profit"] = request.take_profit
            if request.stop_loss:
                params["stop_loss"] = request.stop_loss

        data = await self._request(
            "POST", "/v2/orders", json_data=params, weight=1
        )

        return self._parse_order_response(data)

    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancela una orden específica por ID."""
        result = await self._request("DELETE", f"/v2/orders/{order_id}", weight=1)
        # DELETE returns 204 No Content, fetch the order to get final status
        return await self.get_order(order_id)

    async def cancel_order_by_client_id(self, client_order_id: str) -> OrderResponse:
        """Cancela una orden por client_order_id."""
        result = await self._request("DELETE", "/v2/orders", params={"client_order_id": client_order_id}, weight=1)
        # Fetch the cancelled order
        return await self.get_order_by_client_id(client_order_id)

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancela todas las órdenes abiertas."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()
        
        # Use a custom request to handle 207 Multi-Status
        await self._rate_limiter.acquire(1)
        await self._ensure_session()
        url = f"{self.base_url}/v2/orders"
        
        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }
        
        last_exception = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                async with self._session.delete(
                    url, params=params, headers=headers
                ) as resp:
                    latency = time.perf_counter() - start
                    self._latencies.append(latency)
                    
                    if resp.status in (200, 207):
                        self._request_count += 1
                        data = await resp.json()
                        if isinstance(data, list):
                            return len(data)
                        return 0
                    elif resp.status == 429:
                        retry_after = int(resp.headers.get("Retry-After", 1))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    elif resp.status == 401:
                        error_text = await resp.text()
                        raise RuntimeError(f"Authentication failed: {error_text}")
                    elif resp.status == 403:
                        error_text = await resp.text()
                        raise RuntimeError(f"Forbidden: {error_text}")
                    else:
                        error_text = await resp.text()
                        raise RuntimeError(f"HTTP {resp.status}: {error_text}")
                        
            except asyncio.TimeoutError:
                last_exception = asyncio.TimeoutError("Request timeout")
            except aiohttp.ClientError as e:
                last_exception = e
            
            if attempt < self.max_retries:
                delay = min(self.base_retry_delay * (2 ** attempt), self.max_retry_delay)
                await asyncio.sleep(delay)
        
        self._error_count += 1
        raise RuntimeError(f"Request failed after {self.max_retries + 1} attempts: {last_exception}")

    async def get_orders(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        after: Optional[str] = None,
        until: Optional[str] = None,
        direction: str = "desc",
        nested: bool = False,
    ) -> List[OrderResponse]:
        """Obtiene lista de órdenes."""
        params = {
            "limit": limit,
            "direction": direction,
            "nested": "true" if nested else "false",
        }
        if status:
            params["status"] = status
        if after:
            params["after"] = after
        if until:
            params["until"] = until

        data = await self._request("GET", "/v2/orders", params=params, weight=1)

        orders = []
        for o in data:
            orders.append(self._parse_order_response(o))
        return orders

    async def get_order(self, order_id: str) -> OrderResponse:
        """Obtiene una orden por ID."""
        data = await self._request("GET", f"/v2/orders/{order_id}", weight=1)
        return self._parse_order_response(data)

    async def get_order_by_client_id(self, client_order_id: str) -> OrderResponse:
        """Obtiene una orden por client_order_id."""
        params = {"client_order_id": client_order_id}
        data = await self._request("GET", "/v2/orders", params=params, weight=1)
        if isinstance(data, list) and data:
            return self._parse_order_response(data[0])
        raise ValueError(f"Order with client_order_id {client_order_id} not found")

    async def replace_order(
        self,
        order_id: str,
        qty: Optional[float] = None,
        time_in_force: Optional[str] = None,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Reemplaza una orden existente."""
        params = {}
        if qty is not None:
            params["qty"] = str(qty)
        if time_in_force:
            params["time_in_force"] = time_in_force
        if limit_price is not None:
            params["limit_price"] = str(limit_price)
        if stop_price is not None:
            params["stop_price"] = str(stop_price)
        if client_order_id:
            params["client_order_id"] = client_order_id

        data = await self._request("PATCH", f"/v2/orders/{order_id}", json_data=params, weight=1)
        return self._parse_order_response(data)

    # --- Market Data (via Alpaca Data API) ---

    async def get_latest_quote(self, symbol: str) -> Dict:
        """Obtiene último quote para un símbolo."""
        return await self._request(
            "GET",
            f"/stocks/{symbol.upper()}/quotes/latest",
            params={"feed": self.data_feed},
            weight=1,
            use_data_api=True,
        )

    async def get_latest_trade(self, symbol: str) -> Dict:
        """Obtiene último trade para un símbolo."""
        return await self._request(
            "GET",
            f"/stocks/{symbol.upper()}/trades/latest",
            params={"feed": self.data_feed},
            weight=1,
            use_data_api=True,
        )

    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Min",
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 1000,
    ) -> Dict:
        """Obtiene barras históricas.

        Nota: el feed gratuito ``iex`` limita el historial reciente
        (p. ej. ~9 barras 1H). Con mercado cerrado o límites del plan,
        complementar con ``data/market/<SYM>_<res>.csv`` (HistoricalCsvFeed).
        """
        params = {
            "timeframe": timeframe,
            "limit": limit,
            "feed": self.data_feed,
        }
        if start:
            params["start"] = start
        if end:
            params["end"] = end

        return await self._request(
            "GET",
            f"/stocks/{symbol.upper()}/bars",
            params=params,
            weight=1,
            use_data_api=True,
        )

    # --- Clock / Calendar ---

    async def get_clock(self) -> Dict:
        """Obtiene reloj del mercado (estado: open/closed)."""
        return await self._request("GET", "/v2/clock", weight=1)

    async def get_calendar(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> List[Dict]:
        """Obtiene calendario de trading."""
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return await self._request("GET", "/v2/calendar", params=params, weight=1)

    # --- Assets ---

    async def get_assets(
        self,
        status: str = "active",
        asset_class: str = "us_equity",
    ) -> List[Dict]:
        """Obtiene lista de assets."""
        params = {"status": status, "asset_class": asset_class}
        return await self._request("GET", "/v2/assets", params=params, weight=1)

    async def get_asset(self, symbol: str) -> Dict:
        """Obtiene info de un asset específico."""
        return await self._request("GET", f"/v2/assets/{symbol.upper()}", weight=1)

    # --- Helpers ---

    def _parse_order_response(self, data: Dict) -> OrderResponse:
        return OrderResponse(
            id=data.get("id", ""),
            client_order_id=data.get("client_order_id", ""),
            symbol=data.get("symbol", ""),
            status=data.get("status", ""),
            side=data.get("side", ""),
            order_type=data.get("order_type", ""),
            qty=float(data.get("qty", 0)),
            price=float(data.get("limit_price", 0)) if data.get("limit_price") else None,
            stop_price=float(data.get("stop_price", 0)) if data.get("stop_price") else None,
            filled_qty=float(data.get("filled_qty", 0)),
            filled_avg_price=float(data.get("filled_avg_price", 0)) if data.get("filled_avg_price") else None,
            commission=float(data.get("commission", 0)) if data.get("commission") else 0.0,
            created_at=data.get("created_at", 0),
            updated_at=data.get("updated_at", 0),
            raw=data,
        )

    # --- Convenience Methods ---

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        time_in_force: str = "day",
        extended_hours: bool = False,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Orden MARKET simple."""
        return await self.submit_order(OrderRequest(
            symbol=symbol,
            side=side,
            order_type=self.ORDER_TYPE_MARKET,
            qty=qty,
            time_in_force=time_in_force,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
        ))

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        time_in_force: str = "day",
        extended_hours: bool = False,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Orden LIMIT simple."""
        return await self.submit_order(OrderRequest(
            symbol=symbol,
            side=side,
            order_type=self.ORDER_TYPE_LIMIT,
            qty=qty,
            price=price,
            time_in_force=time_in_force,
            extended_hours=extended_hours,
            client_order_id=client_order_id,
        ))

    async def place_stop_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Orden STOP simple."""
        return await self.submit_order(OrderRequest(
            symbol=symbol,
            side=side,
            order_type=self.ORDER_TYPE_STOP,
            qty=qty,
            stop_price=stop_price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        ))

    async def place_bracket_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        take_profit_price: float,
        stop_loss_price: float,
        limit_price: Optional[float] = None,
        time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Orden BRACKET (entrada + take profit + stop loss)."""
        order_type = self.ORDER_TYPE_LIMIT if limit_price else self.ORDER_TYPE_MARKET
        return await self.submit_order(OrderRequest(
            symbol=symbol,
            side=side,
            order_type=order_type,
            qty=qty,
            price=limit_price,
            time_in_force=time_in_force,
            order_class="bracket",
            take_profit={"limit_price": str(take_profit_price)},
            stop_loss={"stop_price": str(stop_loss_price)},
            client_order_id=client_order_id,
        ))

    # --- Métricas ---

    def get_metrics(self) -> Dict:
        return {
            "request_count": self._request_count,
            "error_count": self._error_count,
            "error_rate": self._error_count / max(1, self._request_count),
            "avg_latency_ms": np.mean(self._latencies) * 1000 if self._latencies else 0,
            "p50_latency_ms": np.percentile(self._latencies, 50) * 1000 if self._latencies else 0,
            "p99_latency_ms": np.percentile(self._latencies, 99) * 1000 if self._latencies else 0,
            "session_closed": self._closed,
        }


# Alias para compatibilidad con BrokerClientProtocol
BrokerClientProtocol = AlpacaOrderClient