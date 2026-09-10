"""Account, Positions, and Market Data client for Alpaca Order Client."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client_transport import AlpacaOrderTransport


class AlpacaMarketMixin(AlpacaOrderTransport):
    """Mixin que agrupa operaciones de cuenta, posiciones y datos de mercado."""

    async def get_account(self) -> Dict[str, Any]:
        """Obtiene información de la cuenta."""
        return await self._request("GET", "/v2/account", weight=1)

    async def get_account_configurations(self) -> Dict[str, Any]:
        """Obtiene configuraciones de la cuenta."""
        return await self._request("GET", "/v2/account/configurations", weight=1)

    async def get_positions(self) -> List[Dict[str, Any]]:
        """Obtiene todas las posiciones abiertas."""
        res = await self._request("GET", "/v2/positions", weight=1)
        return cast(List[Dict[str, Any]], res)

    async def get_position(self, symbol: str) -> Dict[str, Any]:
        """Obtiene posición para un símbolo específico. Retorna qty=0 si no existe."""
        try:
            return await self._request("GET", f"/v2/positions/{symbol.upper()}", weight=1)
        except RuntimeError as e:
            if "404" in str(e) or "position does not exist" in str(e):
                return {"symbol": symbol.upper(), "qty": "0", "side": "long", "avg_entry_price": "0", "market_value": "0"}
            raise

    async def close_position(self, symbol: str) -> Dict[str, Any]:
        """Cierra una posición específica a mercado."""
        try:
            return await self._request("DELETE", f"/v2/positions/{symbol.upper()}", weight=1)
        except RuntimeError as e:
            if "404" in str(e) or "position does not exist" in str(e):
                return {"symbol": symbol.upper(), "status": "closed", "qty": "0"}
            raise

    async def close_all_positions(self, cancel_orders: bool = True) -> List[Dict[str, Any]]:
        """Cierra todas las posiciones."""
        params = {"cancel_orders": "true" if cancel_orders else "false"}
        res = await self._request("DELETE", "/v2/positions", params=params, weight=1)
        return cast(List[Dict[str, Any]], res)

    async def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Obtiene último quote para un símbolo."""
        return await self._request(
            "GET", f"/stocks/{symbol.upper()}/quotes/latest",
            params={"feed": self.data_feed}, weight=1, use_data_api=True,
        )

    async def get_latest_trade(self, symbol: str) -> Dict[str, Any]:
        """Obtiene último trade para un símbolo."""
        return await self._request(
            "GET", f"/stocks/{symbol.upper()}/trades/latest",
            params={"feed": self.data_feed}, weight=1, use_data_api=True,
        )

    async def get_bars(
        self, symbol: str, timeframe: str = "1Min",
        start: Optional[str] = None, end: Optional[str] = None, limit: int = 1000,
    ) -> Dict[str, Any]:
        """Obtiene barras históricas."""
        params: Dict[str, Any] = {"timeframe": timeframe, "limit": limit, "feed": self.data_feed}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        return await self._request("GET", f"/stocks/{symbol.upper()}/bars", params=params, weight=1, use_data_api=True)

    async def get_clock(self) -> Dict[str, Any]:
        """Obtiene reloj del mercado (estado: open/closed)."""
        return await self._request("GET", "/v2/clock", weight=1)

    async def get_calendar(self, start: Optional[str] = None, end: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene calendario de trading."""
        params = {}
        if start:
            params["start"] = start
        if end:
            params["end"] = end
        res = await self._request("GET", "/v2/calendar", params=params, weight=1)
        return cast(List[Dict[str, Any]], res)

    async def get_assets(self, status: str = "active", asset_class: str = "us_equity") -> List[Dict[str, Any]]:
        """Obtiene lista de assets."""
        params = {"status": status, "asset_class": asset_class}
        res = await self._request("GET", "/v2/assets", params=params, weight=1)
        return cast(List[Dict[str, Any]], res)

    async def get_asset(self, symbol: str) -> Dict[str, Any]:
        """Obtiene info de un asset específico."""
        return await self._request("GET", f"/v2/assets/{symbol.upper()}", weight=1)
