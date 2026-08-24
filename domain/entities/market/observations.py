"""Observaciones de mercado — dominio ZENIN Market.

Contract v1 — observaciones inmutables y fuertemente tipadas.
El dominio no conoce infraestructura (sqlalchemy, pymysql, providers):
solo tipos puros de Python.

Timestamp: epoch en segundos con sub-segundo (float). Para las fuentes
con ``NANOSECOND_TIMESTAMP`` la precisión queda conservada por el float;
la persistencia usará ``DATETIME(6)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .data_status import DataStatus
from .validators import (
    validate_price,
    validate_size,
    validate_symbol,
    validate_timestamp,
)

_Price = float
_Size = float


@dataclass(frozen=True, slots=True, kw_only=True)
class MarketObservation:
    """Base inmutable de toda observación de mercado.

    Attributes:
        symbol: Símbolo/instrumento (ej: "NVDA", "BTC/USD").
        timestamp: Epoch en segundos con precisión sub-segundo.
        data_status: Estado de frescura/origen de la observación.
        source_provider: Proveedor de origen (ej: "alpaca", "binance").
        venue: Venue/exchange si el proveedor lo reporta (``None`` si no).
    """

    symbol: str
    timestamp: float
    data_status: DataStatus
    source_provider: str
    venue: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", validate_symbol(self.symbol))
        validate_timestamp(self.timestamp)
        if not isinstance(self.data_status, DataStatus):
            raise TypeError(
                f"data_status debe ser DataStatus, no {self.data_status!r}"
            )
        provider = self.source_provider.strip()
        if not provider:
            raise ValueError("source_provider no puede ser vacío")
        object.__setattr__(self, "source_provider", provider)
        if self.venue is not None:
            venue = self.venue.strip()
            if not venue:
                raise ValueError("venue no puede ser vacío")
            object.__setattr__(self, "venue", venue)

    @property
    def is_live(self) -> bool:
        """``True`` si la observación es una señal live."""
        return self.data_status.is_live_signal


@dataclass(frozen=True, slots=True, kw_only=True)
class Trade(MarketObservation):
    """Operación ejecutada (tick de transacción).

    Attributes:
        price: Precio de la operación.
        size: Cantidad operada.
        trade_id: ID de la operación si el proveedor lo reporta.
        taker_side: Lado agresor ("buy"/"sell") — cripto; ``None`` si no.
        conditions: Condiciones de tape si el proveedor las reporta.
        tape: Cinta consolidada ("A"/"B"/"C") si aplica.
        corrected: ``True`` si el proveedor marcó cancel/corrección.
    """

    price: float
    size: float
    trade_id: str | None = None
    taker_side: str | None = None
    conditions: tuple[str, ...] = field(default_factory=tuple)
    tape: str | None = None
    corrected: bool = False

    def __post_init__(self) -> None:
        MarketObservation.__post_init__(self)
        validate_price(self.price, "price")
        validate_size(self.size, "size")
        if self.size == 0:
            raise ValueError("size no puede ser 0 en un Trade")
        if self.conditions:
            for cond in self.conditions:
                if not isinstance(cond, str) or not cond:
                    raise TypeError("conditions debe ser tupla de str no vacías")
        if self.taker_side is not None and self.taker_side not in ("buy", "sell"):
            raise ValueError(f"taker_side inválido: {self.taker_side!r}")


@dataclass(frozen=True, slots=True, kw_only=True)
class Quote(MarketObservation):
    """Mejor bid/ask (top-of-book) para un instrumento.

    Nota: no se impone bid <= ask porque libros cruzados existen
    temporalmente en feeds reales; la validación rechaza estados
    imposibles (no finitos, <= 0), no estados que los venues reportan.
    """

    bid: float
    bid_size: float
    ask: float
    ask_size: float
    bid_exchange: str | None = None
    ask_exchange: str | None = None
    conditions: tuple[str, ...] = field(default_factory=tuple)
    tape: str | None = None

    def __post_init__(self) -> None:
        MarketObservation.__post_init__(self)
        validate_price(self.bid, "bid")
        validate_price(self.ask, "ask")
        validate_size(self.bid_size, "bid_size")
        validate_size(self.ask_size, "ask_size")

    @property
    def spread(self) -> float:
        """Diferencia ask - bid (puede ser negativa en libros cruzados)."""
        return self.ask - self.bid

    @property
    def midpoint(self) -> float:
        """Punto medio bid/ask."""
        return (self.bid + self.ask) / 2.0


@dataclass(frozen=True, slots=True, kw_only=True)
class Candle(MarketObservation):
    """Vela OHLCV (propia del dominio o del provider).

    Attributes:
        open/high/low/close: Precios OHLC (close puede ser <= 0 en casos
            degenerados? No: siempre > 0 en mercados reales).
        volume: Volumen de la vela (>= 0).
        vwap: Precio promedio ponderado por volumen si está disponible.
        trade_count: Número de operaciones que compusieron la vela.
        interval_seconds: Duración de la vela en segundos.
        adjusted: ``True`` si los precios están ajustados por splits/dividendos.
    """

    open: float
    high: float
    low: float
    close: float
    volume: float
    interval_seconds: int
    vwap: float | None = None
    trade_count: int | None = None
    adjusted: bool = False

    def __post_init__(self) -> None:
        MarketObservation.__post_init__(self)
        validate_price(self.open, "open")
        validate_price(self.high, "high")
        validate_price(self.low, "low")
        validate_price(self.close, "close")
        validate_size(self.volume, "volume")
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds debe ser > 0")
        if self.low > min(self.open, self.close):
            raise ValueError(f"low > min(open, close): low={self.low}")
        if self.high < max(self.open, self.close):
            raise ValueError(f"high < max(open, close): high={self.high}")
        if self.vwap is not None:
            validate_price(self.vwap, "vwap")
        if self.trade_count is not None and self.trade_count < 0:
            raise ValueError("trade_count debe ser >= 0")

    @property
    def body(self) -> float:
        """Diferencia close - open."""
        return self.close - self.open

    @property
    def is_bullish(self) -> bool:
        """``True`` si la vela cerró por encima del open."""
        return self.close > self.open


@dataclass(frozen=True, slots=True, kw_only=True)
class OrderBookSnapshot(MarketObservation):
    """Snapshot del libro de órdenes (solo si el provider tiene L2).

    Attributes:
        bids: Niveles (price, size) ordenados descendente por precio.
        asks: Niveles (price, size) ordenados ascendente por precio.
        reset: ``True`` si el snapshot reemplaza el libro local completo.
    """

    bids: tuple[tuple[float, float], ...]
    asks: tuple[tuple[float, float], ...]
    reset: bool = False

    def __post_init__(self) -> None:
        MarketObservation.__post_init__(self)
        bids = tuple(self.bids)
        asks = tuple(self.asks)
        if not bids and not asks:
            raise ValueError("OrderBookSnapshot no puede tener libro vacío")
        for level in bids:
            validate_price(level[0], "bid price")
            validate_size(level[1], "bid size")
        for level in asks:
            validate_price(level[0], "ask price")
            validate_size(level[1], "ask size")
        self._validate_ordering(bids, descending=True, name="bids")
        self._validate_ordering(asks, descending=False, name="asks")
        object.__setattr__(self, "bids", bids)
        object.__setattr__(self, "asks", asks)

    @staticmethod
    def _validate_ordering(
        levels: tuple[tuple[float, float], ...],
        *,
        descending: bool,
        name: str,
    ) -> None:
        for prev, curr in zip(levels, levels[1:], strict=False):
            if descending and curr[0] >= prev[0]:
                raise ValueError(
                    f"{name} debe estar ordenado descendente por precio: "
                    f"{prev[0]} -> {curr[0]}"
                )
            if not descending and curr[0] <= prev[0]:
                raise ValueError(
                    f"{name} debe estar ordenado ascendente por precio: "
                    f"{prev[0]} -> {curr[0]}"
                )

    @property
    def best_bid(self) -> float | None:
        """Mejor precio bid."""
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> float | None:
        """Mejor precio ask."""
        return self.asks[0][0] if self.asks else None

    @property
    def imbalance(self) -> float:
        """Desbalance de volumen bid vs ask en (-1, 1); 0 si no hay datos."""
        bid_vol = sum(level[1] for level in self.bids)
        ask_vol = sum(level[1] for level in self.asks)
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total
