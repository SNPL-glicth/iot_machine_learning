"""Adaptadores de proveedores de datos de mercado (FASE 4, 5 y 6).

FASE 4: convierten payloads de los proveedores (Alpaca/Binance) a
entidades del dominio ZENIN. Sin conexiones, sin claves API, sin
estado: la conversión es pura y testeada con fixtures congelados.

FASE 5: feeds históricos (CSV congelado en disco) para el Market
Replay; el replay jamás toca la red.

FASE 6: LiveFeed wrapper con detección de gaps y estados de conexión;
LiveShadowRunner para el modo shadow (sin persistencia, consola).
"""

from .alpaca_adapter import ALPACA_PROFILE, AlpacaAdapter
from .binance_adapter import BINANCE_PROFILE, BinanceAdapter
from .csv_feed import HistoricalCsvFeed
from .live_feed import GapDetected, LiveFeed, StateTransition
from .live_fragment import (
    RESOLUTIONS,
    DropWindowsFeed,
    FragmentFeed,
    drop_windows,
    fmt_ts,
    fragment_bounds,
    parse_drop,
)
from .live_shadow import DegradedWindow, LiveShadowResult, LiveShadowRunner

__all__ = [
    "ALPACA_PROFILE",
    "AlpacaAdapter",
    "BINANCE_PROFILE",
    "BinanceAdapter",
    "HistoricalCsvFeed",
    "GapDetected",
    "LiveFeed",
    "StateTransition",
    "DegradedWindow",
    "LiveShadowResult",
    "LiveShadowRunner",
    "RESOLUTIONS",
    "FragmentFeed",
    "DropWindowsFeed",
    "fragment_bounds",
    "parse_drop",
    "drop_windows",
    "fmt_ts",
]
