"""Observaciones de mercado — dominio ZENIN Market.

Contract v1 — observaciones inmutables y fuertemente tipadas.
El dominio no conoce infraestructura (sqlalchemy, pymysql, providers):
solo tipos puros de Python.

Timestamp: epoch en segundos con sub-segundo (float). Para las fuentes
con ``NANOSECOND_TIMESTAMP`` la precisión queda conservada por el float;
la persistencia usará ``DATETIME(6)``.
"""

from .market_observation import MarketObservation
from .trade import Trade
from .quote import Quote
from .candle import Candle
from .order_book import OrderBookSnapshot

__all__ = [
    "MarketObservation",
    "Trade",
    "Quote",
    "Candle",
    "OrderBookSnapshot",
]