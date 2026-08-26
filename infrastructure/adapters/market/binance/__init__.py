"""Binance adapters — WebSocket feed, REST client, order book state, account management.

FASE 7: Componentes para trading live en Binance:
- ws_client: WebSocket client con reconexión exponencial
- ws_feed: Feed asíncrono con order book L2 sincronizado
- order_book_state: OrderBookL2 con sincronización snapshot+deltas
- order_client: Cliente REST firmado para ejecución de órdenes
- account: Gestión de cuenta y posiciones
"""

from .ws_client import BinanceWSClient, create_market_streams, ConnectionState
from .ws_feed import BinanceWSFeed, BinanceLiveFeed, FeedStats
from .order_book_state import OrderBookL2, OrderBookMetrics, PriceLevel
from .order_client import BinanceOrderClient, OrderRequest, OrderResponse, RateLimiter
from .account import BinanceAccount, Position, AccountSnapshot

__all__ = [
    "BinanceWSClient",
    "create_market_streams",
    "ConnectionState",
    "BinanceWSFeed",
    "BinanceLiveFeed",
    "FeedStats",
    "OrderBookL2",
    "OrderBookMetrics",
    "PriceLevel",
    "BinanceOrderClient",
    "OrderRequest",
    "OrderResponse",
    "RateLimiter",
    "BinanceAccount",
    "Position",
    "AccountSnapshot",
]