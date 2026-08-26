"""Adaptadores de proveedores de datos de mercado (FASE 4, 5, 6 y 7).

FASE 4: convierten payloads de los proveedores (Alpaca/Binance) a
entidades del dominio ZENIN. Sin conexiones, sin claves API, sin
estado: la conversión es pura y testeada con fixtures congelados.

FASE 5: feeds históricos (CSV congelado en disco) para el Market
Replay; el replay jamás toca la red.

FASE 6: LiveFeed wrapper con detección de gaps y estados de conexión;
LiveShadowRunner para el modo shadow (sin persistencia, consola).

FASE 7: BinanceWSFeed event-driven con order book L2 sincronizado,
feature extraction y ejecución de órdenes live.
"""

from .alpaca_adapter import ALPACA_PROFILE, AlpacaAdapter
from .binance_adapter import BINANCE_PROFILE, BinanceAdapter
from .binance.ws_client import BinanceWSClient, create_market_streams, ConnectionState
from .binance.ws_feed import BinanceWSFeed, FeedStats
from .binance.order_book_state import OrderBookL2, OrderBookMetrics, PriceLevel
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
    "BinanceWSClient",
    "create_market_streams",
    "ConnectionState",
    "BinanceWSFeed",
    "BinanceLiveFeed",
    "FeedStats",
    "OrderBookL2",
    "OrderBookMetrics",
    "PriceLevel",
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
