"""BinanceOrderClient -- Cliente REST firmado para Binance Futures.

Implementa BrokerClientProtocol para el execution handler.
Rate limiting, retry con backoff, order types: LIMIT GTX, MARKET, STOP, TAKE_PROFIT.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import deque

import aiohttp
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OrderRequest:
    """Solicitud de orden estandarizada."""
    symbol: str
    side: str              # "BUY" o "SELL"
    order_type: str        # "LIMIT", "MARKET", "STOP_MARKET", "TAKE_PROFIT_MARKET"
    qty: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK, GTX
    reduce_only: bool = False
    client_order_id: Optional[str] = None


@dataclass
class OrderResponse:
    """Respuesta de orden de Binance."""
    order_id: int
    client_order_id: str
    symbol: str
    status: str           # NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED, EXPIRED
    side: str
    order_type: str
    qty: float
    price: Optional[float]
    stop_price: Optional[float]
    executed_qty: float = 0.0
    avg_price: float = 0.0
    commission: float = 0.0
    commission_asset: str = ""
    timestamp: float = 0.0
    raw: Dict = field(default_factory=dict)


class RateLimiter:
    """Rate limiter token bucket para Binance API."""

    def __init__(
        self,
        max_requests_per_minute: int = 1200,
        max_orders_per_second: int = 10,
        max_orders_per_day: int = 200000,
    ):
        self.max_rpm = max_requests_per_minute
        self.max_ops = max_orders_per_second
        self.max_daily = max_orders_per_day

        self._request_times: deque = deque(maxlen=max_requests_per_minute)
        self._order_times: deque = deque(maxlen=max_orders_per_second)
        self._daily_count = 0
        self._day_start = time.time()

        self._lock = asyncio.Lock()

    async def acquire(self, weight: int = 1) -> None:
        """Espera hasta que haya capacidad."""
        async with self._lock:
            now = time.time()

            # Reset daily counter
            if now - self._day_start >= 86400:
                self._daily_count = 0
                self._day_start = now

            if self._daily_count >= self._daily:
                wait = 86400 - (now - self._day_start)
                logger.warning(f"Daily rate limit reached, waiting {wait:.0f}s")
                await asyncio.sleep(wait)
                self._daily_count = 0
                self._day_start = time.time()

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

            if len(self._order_times) >= self.max_os:
                wait = 1 - (now - self._order_times[0])
                await asyncio.sleep(max(0, wait) + 0.01)

            self._request_times.append(now)
            self._order_times.append(now)
            self._daily_count += 1


class BinanceOrderClient:
    """
    Cliente REST firmado para Binance Futures (USD-M).

    Características:
    - Firma HMAC-SHA256
    - Rate limiting automático
    - Retry con backoff exponencial
    - Tipos de orden: LIMIT (GTX/GTC/IOC/FOK), MARKET, STOP_MARKET, TAKE_PROFIT_MARKET
    - Cancelación individual y masiva
    - Consulta de posiciones y órdenes abiertas
    """

    BASE_URL = "https://fapi.binance.com"
    TESTNET_BASE_URL = "https://testnet.binancefuture.com"

    # Límites de rate limit (oficiales)
    DEFAULT_RPM = 1200      # requests per minute
    DEFAULT_OPS = 10        # orders per second
    DEFAULT_DAILY = 200000  # orders per day

    # Order types
    ORDER_TYPE_LIMIT = "LIMIT"
    ORDER_TYPE_MARKET = "MARKET"
    ORDER_TYPE_STOP_MARKET = "STOP_MARKET"
    ORDER_TYPE_TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"

    # Time in force
    TIME_IN_FORCE_GTC = "GTC"   # Good Till Canceled
    TIME_IN_FORCE_IOC = "IOC"   # Immediate or Cancel
    TIME_IN_FORCE_FOK = "FOK"   # Fill or Kill
    TIME_IN_FORCE_GTX = "GTX"   # Post-Only (Maker only)

    # Order statuses
    STATUS_NEW = "NEW"
    STATUS_PARTIALLY_FILLED = "PARTIALLY_FILLED"
    STATUS_FILLED = "FILLED"
    STATUS_CANCELED = "CANCELED"
    STATUS_REJECTED = "REJECTED"
    STATUS_EXPIRED = "EXPIRED"

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False,
        *,
        base_url: Optional[str] = None,
        recv_window: int = 5000,
        max_retries: int = 3,
        base_retry_delay: float = 0.5,
        max_retry_delay: float = 10.0,
    ):
        """
        Args:
            api_key: API Key de Binance
            api_secret: API Secret de Binance
            testnet: Usar testnet de Binance Futures
            base_url: URL base personalizada (opcional)
            recv_window: Ventana de recepción en ms
            max_retries: Reintentos máximos para requests fallidos
            base_retry_delay: Delay base para backoff exponencial
            max_retry_delay: Delay máximo para reintentos
        """
        self.api_key = api_key
        self.api_secret = api_secret.encode()
        self.testnet = testnet
        self.base_url = base_url or (self.TESTNET_BASE_URL if testnet else self.BASE_URL)
        self.recv_window = recv_window
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
            "BinanceOrderClient initialized",
            extra={"testnet": testnet, "base_url": self.base_url},
        )

    async def __aenter__(self) -> "BinanceOrderClient":
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
                headers={"X-MBX-APIKEY": self.api_key},
            )

    async def close(self) -> None:
        """Cierra la sesión HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()
        self._closed = True

    # --- Core HTTP Methods ---

    def _sign_params(self, params: Dict[str, Any]) -> str:
        """Genera firma HMAC-SHA256 para parámetros."""
        query_string = urllib.parse.urlencode(params, doseq=True)
        signature = hmac.new(
            self.api_secret,
            query_string.encode(),
            hashlib.sha256
        ).hexdigest()
        return signature

    def _get_timestamp(self) -> int:
        return int(time.time() * 1000)

    def _prepare_signed_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara parámetros con timestamp, recvWindow y firma."""
        params = params.copy()
        params["timestamp"] = self._get_timestamp()
        params["recvWindow"] = self.recv_window
        params["signature"] = self._sign_params(params)
        return params

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        signed: bool = False,
        weight: int = 1,
    ) -> Dict:
        """Ejecuta request HTTP con rate limiting, retry y métricas."""
        await self._rate_limiter.acquire(weight)

        await self._ensure_session()
        url = f"{self.base_url}{endpoint}"

        headers = {"X-MBX-APIKEY": self.api_key}
        params = params or {}

        if signed:
            params = self._prepare_signed_params(params)

        # Query string para GET, body para POST
        if method == "GET":
            query = urllib.parse.urlencode(params, doseq=True)
            url = f"{url}?{query}"
            data = None
        else:
            data = params
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        last_exception = None
        for attempt in range(self.max_retries + 1):
            start = time.perf_counter()
            try:
                async with self._session.request(
                    method, url, data=data, headers=headers
                ) as resp:
                    latency = time.perf_counter() - start
                    self._latencies.append(latency)

                    if resp.status == 200:
                        self._request_count += 1
                        return await resp.json()
                    elif resp.status == 429:
                        # Rate limited - esperar y reintentar
                        retry_after = int(resp.headers.get("Retry-After", 1))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    elif resp.status == 418:
                        # IP baneada
                        logger.error("IP banned by Binance")
                        raise RuntimeError("IP banned by Binance")
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
                    10.0,
                )
                await asyncio.sleep(delay)

        self._error_count += 1
        raise RuntimeError(f"Request failed after {self.max_retries + 1} attempts: {last_exception}")

    # --- Public API ---

    async def get_server_time(self) -> int:
        """Obtiene tiempo del servidor."""
        data = await self._request("GET", "/fapi/v1/time", signed=False, weight=1)
        return data["serverTime"]

    async def get_exchange_info(self) -> Dict:
        """Información de intercambio (símbolos, filtros, etc.)."""
        return await self._request("GET", "/fapi/v1/exchangeInfo", signed=False, weight=10)

    async def get_symbol_info(self, symbol: str) -> Dict:
        """Info de un símbolo específico."""
        info = await self.get_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol.upper():
                return s
        raise ValueError(f"Symbol {symbol} not found")

    # --- Account ---

    async def get_account_info(self) -> Dict:
        """Información de cuenta (balance, posiciones, etc.)."""
        return await self._request("GET", "/fapi/v2/account", signed=True, weight=5)

    async def get_balance(self, asset: str = "USDT") -> float:
        """Balance disponible de un asset."""
        account = await self.get_account_info()
        for asset_info in account.get("assets", []):
            if asset_info["asset"] == asset.upper():
                return float(asset_info["availableBalance"])
        return 0.0

    async def get_position_risk(self, symbol: Optional[str] = None) -> List[Dict]:
        """Riesgo de posición (posición actual, PnL no realizado, etc.)."""
        params = {"symbol": symbol.upper()} if symbol else {}
        return await self._request("GET", "/fapi/v2/positionRisk", params, signed=True, weight=5)

    async def get_position(self, symbol: str) -> float:
        """Posición actual (positivo=long, negativo=short)."""
        positions = await self.get_position_risk(symbol)
        for p in positions:
            if p["symbol"] == symbol.upper():
                return float(p["positionAmt"])
        return 0.0

    # --- Orders ---

    async def submit_order(self, request: OrderRequest) -> OrderResponse:
        """Envía orden a Binance."""
        params = {
            "symbol": request.symbol.upper(),
            "side": request.side.upper(),
            "type": request.order_type,
            "quantity": self._format_qty(request.qty),
        }

        if request.order_type in (self.ORDER_TYPE_LIMIT, "STOP", "TAKE_PROFIT"):
            if request.price is None:
                raise ValueError(f"{request.order_type} requires price")
            params["price"] = self._format_price(request.price)

        if request.order_type in ("STOP_MARKET", "TAKE_PROFIT_MARKET"):
            if request.stop_price is None:
                raise ValueError(f"{request.order_type} requires stop_price")
            params["stopPrice"] = self._format_price(request.stop_price)

        if request.time_in_force:
            params["timeInForce"] = request.time_in_force

        if request.reduce_only:
            params["reduceOnly"] = "true"

        if request.client_order_id:
            params["newClientOrderId"] = request.client_order_id

        data = await self._request(
            "POST", "/fapi/v1/order", params, signed=True, weight=1
        )

        return self._parse_order_response(data)

    async def cancel_order(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Cancela una orden específica."""
        params = {"symbol": symbol.upper()}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id

        data = await self._request("DELETE", "/fapi/v1/order", params, signed=True, weight=1)
        return self._parse_order_response(data)

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancela todas las órdenes abiertas (opcionalmente para un símbolo)."""
        params = {}
        if symbol:
            params["symbol"] = symbol.upper()

        data = await self._request("DELETE", "/fapi/v1/allOpenOrders", params, signed=True, weight=1)

        # Respuesta es lista de órdenes canceladas
        if isinstance(data, list):
            return len(data)
        return 0

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[OrderResponse]:
        """Obtiene órdenes abiertas."""
        params = {"symbol": symbol.upper()} if symbol else {}
        data = await self._request("GET", "/fapi/v1/openOrders", params, signed=True, weight=10)

        orders = []
        for o in data:
            orders.append(self._parse_order_response(o))
        return orders

    async def get_order_status(
        self,
        symbol: str,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Consulta estado de una orden."""
        params = {"symbol": symbol.upper()}
        if order_id:
            params["orderId"] = order_id
        if client_order_id:
            params["origClientOrderId"] = client_order_id

        data = await self._request("GET", "/fapi/v1/order", params, signed=True, weight=1)
        return self._parse_order_response(data)

    async def close_position(self, symbol: str) -> OrderResponse:
        """Cierra posición a mercado (reduceOnly)."""
        position = await self.get_position(symbol)
        if position == 0:
            raise ValueError("No open position to close")

        side = "SELL" if position > 0 else "BUY"
        qty = abs(position)

        request = OrderRequest(
            symbol=symbol,
            side=side,
            order_type=self.ORDER_TYPE_MARKET,
            qty=qty,
            reduce_only=True,
            time_in_force=self.TIME_IN_FORCE_IOC,
        )
        return await self.submit_order(request)

    # --- Market Data ---

    async def get_ticker(self, symbol: str) -> Dict:
        """24hr ticker price change statistics."""
        params = {"symbol": symbol.upper()}
        return await self._request("GET", "/fapi/v1/ticker/24hr", params, signed=False, weight=1)

    async def get_mark_price(self, symbol: str) -> float:
        """Mark price actual."""
        params = {"symbol": symbol.upper()}
        data = await self._request("GET", "/fapi/v1/premiumIndex", params, signed=False, weight=1)
        return float(data["markPrice"])

    async def get_funding_rate(self, symbol: str) -> float:
        """Funding rate actual."""
        params = {"symbol": symbol.upper(), "limit": 1}
        data = await self._request("GET", "/fapi/v1/fundingRate", params, signed=False, weight=1)
        return float(data[0]["fundingRate"])

    # --- Helpers ---

    def _format_qty(self, qty: float) -> str:
        """Formatea cantidad según precisión de Binance."""
        # Binance Futures permite hasta 3 decimales para BTCUSDT
        return f"{qty:.3f}"

    def _format_price(self, price: float) -> str:
        """Formatea precio según tick size."""
        # BTCUSDT tick size = 0.1
        return f"{price:.1f}"

    def _parse_order_response(self, data: Dict) -> OrderResponse:
        return OrderResponse(
            order_id=data.get("orderId", 0),
            client_order_id=data.get("clientOrderId", ""),
            symbol=data.get("symbol", ""),
            status=data.get("status", ""),
            side=data.get("side", ""),
            order_type=data.get("type", ""),
            qty=float(data.get("origQty", 0)),
            price=float(data.get("price", 0)) if data.get("price") else None,
            stop_price=float(data.get("stopPrice", 0)) if data.get("stopPrice") else None,
            executed_qty=float(data.get("executedQty", 0)),
            avg_price=float(data.get("avgPrice", 0)) if data.get("avgPrice") else 0.0,
            commission=float(data.get("commission", 0)) if data.get("commission") else 0.0,
            commission_asset=data.get("commissionAsset", ""),
            timestamp=int(data.get("updateTime", time.time() * 1000)) / 1000,
            raw=data,
        )

    # --- Helpers de conveniencia ---

    async def place_limit_post_only(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Orden LIMIT Post-Only (GTX) - solo maker."""
        return await self.submit_order(OrderRequest(
            symbol=symbol,
            side=side,
            order_type=self.ORDER_TYPE_LIMIT,
            qty=qty,
            price=price,
            time_in_force=self.TIME_IN_FORCE_GTX,
            client_order_id=client_order_id,
        ))

    async def place_market(
        self,
        symbol: str,
        side: str,
        qty: float,
        reduce_only: bool = False,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        return await self.submit_order(OrderRequest(
            symbol=symbol,
            side=side,
            order_type=self.ORDER_TYPE_MARKET,
            qty=qty,
            reduce_only=reduce_only,
            time_in_force=self.TIME_IN_FORCE_IOC,
            client_order_id=client_order_id,
        ))

    async def place_stop_loss(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        return await self.submit_order(OrderRequest(
            symbol=symbol,
            side=side,
            order_type=self.ORDER_TYPE_STOP_MARKET,
            qty=qty,
            stop_price=stop_price,
            reduce_only=True,
            client_order_id=client_order_id,
        ))

    async def place_take_profit(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        return await self.submit_order(OrderRequest(
            symbol=symbol,
            side=side,
            order_type=self.ORDER_TYPE_TAKE_PROFIT_MARKET,
            qty=qty,
            stop_price=stop_price,
            reduce_only=True,
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

    async def close(self) -> None:
        """Cierra sesión HTTP."""
        if self._session and not self._session.closed:
            await self._session.close()


# Alias para compatibilidad
BinanceRESTClient = BinanceOrderClient