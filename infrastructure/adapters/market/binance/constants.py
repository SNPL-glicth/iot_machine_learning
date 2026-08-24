"""Binance constants and mappings."""

from __future__ import annotations

from iot_machine_learning.domain.entities.market import Capability, ProviderProfile

__all__ = ["BINANCE_PROFILE", "_KLINE_INTERVALS"]

BINANCE_PROFILE = ProviderProfile(
    provider="binance",
    asset_class="crypto",
    capabilities=frozenset(
        {
            Capability.TRADES,
            Capability.QUOTES,
            Capability.CANDLES,
            Capability.ORDER_BOOK_L2,
            Capability.REALTIME,
            Capability.SECOND_AGGREGATES,
            Capability.VWAP,
        }
    ),
    max_ws_symbols=None,
)

_KLINE_INTERVALS: dict[str, int] = {
    "1s": 1,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "3d": 259200,
    "1w": 604800,
    "1M": 2592000,
}