"""Adaptadores de proveedores de datos de mercado y Zephyr Bot."""

from .zephyr.adapters.alpaca_adapter import ALPACA_PROFILE, AlpacaAdapter
from .zephyr.adapters.binance_adapter import BINANCE_PROFILE, BinanceAdapter
from .binance.ws_client import BinanceWSClient, create_market_streams, ConnectionState
from .binance.ws_feed import BinanceWSFeed, FeedStats
from .binance.order_book_state import OrderBookL2, OrderBookMetrics, PriceLevel
from .alpaca.order_client import AlpacaOrderClient
from .alpaca.account import AlpacaAccount
from .alpaca.ws_feed import AlpacaWSFeed
from .alpaca.order_models import OrderRequest, OrderResponse
from .alpaca.account_models import Position, AccountSnapshot
from .feeds.csv_feed import HistoricalCsvFeed
from .feeds.live_feed import GapDetected, LiveFeed, StateTransition
from .feeds.live_fragment import (
    RESOLUTIONS,
    DropWindowsFeed,
    FragmentFeed,
    drop_windows,
    fmt_ts,
    fragment_bounds,
    parse_drop,
)
from .zephyr.engines.shadow import DegradedWindow, LiveShadowResult, LiveShadowRunner

__all__ = [
    "ALPACA_PROFILE",
    "AlpacaAdapter",
    "BINANCE_PROFILE",
    "BinanceAdapter",
    "BinanceWSClient",
    "create_market_streams",
    "ConnectionState",
    "BinanceWSFeed",
    "FeedStats",
    "OrderBookL2",
    "OrderBookMetrics",
    "PriceLevel",
    "AlpacaOrderClient",
    "AlpacaAccount",
    "AlpacaWSFeed",
    "OrderRequest",
    "OrderResponse",
    "Position",
    "AccountSnapshot",
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
