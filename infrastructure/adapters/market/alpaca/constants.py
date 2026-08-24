"""Alpaca constants and profile."""

from __future__ import annotations

from iot_machine_learning.domain.entities.market import Capability, ProviderProfile

__all__ = ["ALPACA_PROFILE"]

ALPACA_PROFILE = ProviderProfile(
    provider="alpaca",
    asset_class="equities",
    capabilities=frozenset(
        {
            Capability.TRADES,
            Capability.QUOTES,
            Capability.CANDLES,
            Capability.HISTORICAL_TICKS,
            Capability.HISTORICAL_BARS,
            Capability.REALTIME,
            Capability.VWAP,
            Capability.TRADE_CONDITIONS,
            Capability.NANOSECOND_TIMESTAMP,
            Capability.ADJUSTED_SERIES,
        }
    ),
    max_ws_symbols=None,
)