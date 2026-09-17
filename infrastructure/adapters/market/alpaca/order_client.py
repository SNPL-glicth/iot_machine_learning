"""AlpacaOrderClient -- Cliente REST firmado para Alpaca Paper Trading."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client_constants import OrderClientConstants
from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client_market import AlpacaMarketMixin
from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client_transport import AlpacaOrderTransport
from iot_machine_learning.infrastructure.adapters.market.alpaca.order_models import (
    OrderRequest, OrderResponse, parse_order_response,
)
from iot_machine_learning.infrastructure.adapters.market.alpaca.rate_limiter import RateLimiter

import time

logger = logging.getLogger(__name__)

__all__ = ["AlpacaOrderClient", "OrderRequest", "OrderResponse", "RateLimiter", "BrokerClientProtocol"]


class AlpacaOrderClient(OrderClientConstants, AlpacaMarketMixin):
    """Cliente REST firmado para Alpaca Paper Trading (Equities)."""

    def __init__(
        self, api_key: str, api_secret: str, base_url: Optional[str] = None, *,
        data_feed: str = "iex", recv_window: int = 5000, max_retries: int = 3,
        base_retry_delay: float = 0.5, max_retry_delay: float = 10.0,
    ) -> None:
        super().__init__(
            api_key, api_secret, base_url, data_feed=data_feed,
            recv_window=recv_window, max_retries=max_retries,
            base_retry_delay=base_retry_delay, max_retry_delay=max_retry_delay,
        )
        logger.info("AlpacaOrderClient initialized", extra={"base_url": self.base_url, "data_feed": self.data_feed})

    async def __aenter__(self) -> AlpacaOrderClient:
        await self._ensure_session()
        return self

    def _parse_order_response(self, data: Dict[str, Any]) -> OrderResponse:
        return parse_order_response(data)

    async def submit_order(self, request: Optional[OrderRequest] = None, **kwargs: Any) -> OrderResponse:
        """Envía orden a Alpaca con idempotencia y reconciliación de órdenes huérfanas."""
        req = request or OrderRequest(**kwargs)
        params: Dict[str, Any] = {
            "symbol": req.symbol.upper(), "side": req.side.lower(),
            "type": req.order_type, "time_in_force": req.time_in_force,
        }
        if req.notional is not None and req.notional > 0:
            params["notional"] = str(round(req.notional, 2))
        else:
            params["qty"] = str(req.qty)
        if req.order_type in (self.ORDER_TYPE_LIMIT, self.ORDER_TYPE_STOP_LIMIT):
            if req.price is None: raise ValueError(f"{req.order_type} requires price")
            params["limit_price"] = str(req.price)
        if req.order_type in (self.ORDER_TYPE_STOP, self.ORDER_TYPE_STOP_LIMIT):
            if req.stop_price is None: raise ValueError(f"{req.order_type} requires stop_price")
            params["stop_price"] = str(req.stop_price)
        if req.order_type == self.ORDER_TYPE_TRAILING_STOP:
            if req.trail_price is None and req.trail_percent is None: raise ValueError("trailing_stop requires trail")
            if req.trail_price is not None: params["trail_price"] = str(req.trail_price)
            if req.trail_percent is not None: params["trail_percent"] = str(req.trail_percent)
        if req.extended_hours: params["extended_hours"] = True
        cid = req.client_order_id or f"zeph_{req.symbol.lower()}_{int(time.time()*1000)}"
        params["client_order_id"] = cid
        if req.order_class:
            params["order_class"] = req.order_class
            if req.take_profit: params["take_profit"] = req.take_profit
            if req.stop_loss: params["stop_loss"] = req.stop_loss

        try:
            data = await self._request("POST", "/v2/orders", json_data=params, weight=1)
            return self._parse_order_response(data)
        except RuntimeError as exc:
            try:
                existing = await self.get_order_by_client_id(cid)
                if existing:
                    logger.info("Reconciled orphan order: order %s was confirmed on exchange", existing.id)
                    return existing
            except Exception:
                pass
            raise


    async def cancel_order(self, order_id: str) -> OrderResponse:
        """Cancela una orden específica por ID."""
        try: await self._request("DELETE", f"/v2/orders/{order_id}", weight=1)
        except RuntimeError as e:
            if "422" not in str(e): raise
        return await self.get_order(order_id)

    async def cancel_order_by_client_id(self, client_order_id: str) -> OrderResponse:
        """Cancela una orden por client_order_id."""
        try: await self._request("DELETE", "/v2/orders", params={"client_order_id": client_order_id}, weight=1)
        except RuntimeError as e:
            if "422" not in str(e): raise
        return await self.get_order_by_client_id(client_order_id)

    async def cancel_all_orders(self, symbol: Optional[str] = None) -> int:
        """Cancela órdenes abiertas (filtradas por símbolo si se especifica)."""
        if symbol:
            sym_u = symbol.upper()
            open_orders = await self.get_orders(status="open")
            matched = [o for o in open_orders if o.symbol.upper() == sym_u]
            cancelled_ids: List[str] = []
            failed_ids: List[str] = []
            for o in matched:
                try:
                    await self.cancel_order(o.id)
                    cancelled_ids.append(o.id)
                except Exception as e:
                    logger.warning("Failed to cancel order %s for %s: %s", o.id, sym_u, e)
                    failed_ids.append(o.id)
            if failed_ids:
                logger.error(
                    "cancel_all_orders for %s partially failed: cancelled=%d, failed=%d (failed_ids=%s)",
                    sym_u, len(cancelled_ids), len(failed_ids), failed_ids,
                )
            return len(cancelled_ids)
        data = await self._request("DELETE", "/v2/orders", weight=1)
        if isinstance(data, list):
            return sum(1 for item in data if isinstance(item, dict) and item.get("status", 200) < 400)
        return 0

    async def get_orders(
        self, status: Optional[str] = None, limit: int = 100,
        after: Optional[str] = None, until: Optional[str] = None,
        direction: str = "desc", nested: bool = False,
    ) -> List[OrderResponse]:
        """Obtiene lista de órdenes."""
        params: Dict[str, Any] = {"limit": limit, "direction": direction, "nested": "true" if nested else "false"}
        if status: params["status"] = status
        if after: params["after"] = after
        if until: params["until"] = until
        data: Any = await self._request("GET", "/v2/orders", params=params, weight=1)
        raw_list: List[Dict[str, Any]] = data if isinstance(data, list) else []
        return [self._parse_order_response(o) for o in raw_list]

    async def get_order(self, order_id: str) -> OrderResponse:
        """Obtiene una orden por ID."""
        data = await self._request("GET", f"/v2/orders/{order_id}", weight=1)
        return self._parse_order_response(data)

    async def get_order_by_client_id(self, client_order_id: str) -> OrderResponse:
        """Obtiene una orden por client_order_id."""
        data: Any = await self._request("GET", "/v2/orders", params={"client_order_id": client_order_id}, weight=1)
        if isinstance(data, list) and data:
            return self._parse_order_response(data[0])
        raise ValueError(f"Order with client_order_id {client_order_id} not found")

    async def replace_order(
        self, order_id: str, qty: Optional[float] = None, time_in_force: Optional[str] = None,
        limit_price: Optional[float] = None, stop_price: Optional[float] = None, client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Reemplaza una orden existente."""
        params: Dict[str, Any] = {k: str(v) for k, v in [("qty", qty), ("time_in_force", time_in_force), ("limit_price", limit_price), ("stop_price", stop_price), ("client_order_id", client_order_id)] if v is not None}
        data = await self._request("PATCH", f"/v2/orders/{order_id}", json_data=params, weight=1)
        return self._parse_order_response(data)

    async def place_market_order(
        self, symbol: str, side: str, qty: float, time_in_force: str = "day",
        extended_hours: bool = False, client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Orden MARKET simple."""
        return await self.submit_order(OrderRequest(
            symbol=symbol, side=side, order_type=self.ORDER_TYPE_MARKET, qty=qty,
            time_in_force=time_in_force, extended_hours=extended_hours, client_order_id=client_order_id,
        ))

    async def place_limit_order(
        self, symbol: str, side: str, qty: float, price: float, time_in_force: str = "day",
        extended_hours: bool = False, client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Orden LIMIT simple."""
        return await self.submit_order(OrderRequest(
            symbol=symbol, side=side, order_type=self.ORDER_TYPE_LIMIT, qty=qty, price=price,
            time_in_force=time_in_force, extended_hours=extended_hours, client_order_id=client_order_id,
        ))

    async def place_stop_order(
        self, symbol: str, side: str, qty: float, stop_price: float, time_in_force: str = "day",
        client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Orden STOP simple."""
        return await self.submit_order(OrderRequest(
            symbol=symbol, side=side, order_type=self.ORDER_TYPE_STOP, qty=qty, stop_price=stop_price,
            time_in_force=time_in_force, client_order_id=client_order_id,
        ))

    async def place_bracket_order(
        self, symbol: str, side: str, qty: float, take_profit_price: float, stop_loss_price: float,
        limit_price: Optional[float] = None, time_in_force: str = "day", client_order_id: Optional[str] = None,
    ) -> OrderResponse:
        """Orden BRACKET (entrada + take profit + stop loss)."""
        order_type = self.ORDER_TYPE_LIMIT if limit_price else self.ORDER_TYPE_MARKET
        return await self.submit_order(OrderRequest(
            symbol=symbol, side=side, order_type=order_type, qty=qty, price=limit_price,
            time_in_force=time_in_force, order_class="bracket",
            take_profit={"limit_price": str(take_profit_price)}, stop_loss={"stop_price": str(stop_loss_price)},
            client_order_id=client_order_id,
        ))


# Alias para compatibilidad con BrokerClientProtocol
BrokerClientProtocol = AlpacaOrderClient