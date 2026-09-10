"""Features L1 de microestructura (FASE 3, L1-only por decisión).

Qué comportamiento de las órdenes precede estadísticamente a los
movimientos: spread, imbalance top-of-book, OFI (Cont–Kukanov–Stoikov
simplificado a L1), agresión compradora/vendedora (taker side),
intensidad, impacto y calidad (libros cruzados).

Puro y determinista: mismos quotes/trades → mismas features.
Sin L2, sin inferencia deGBP: el taker_side lo da el provider.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..observations import Quote, Trade

__all__ = ["L1Features", "l1_features", "MIN_QUOTES"]


MIN_QUOTES: int = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class L1Features:
    """Vector microestructural de un tramo (top-of-book + tape)."""

    n_quotes: int
    n_trades: int
    span_seconds: float
    spread_bps: float  # (ask-bid)/mid·1e4 del último quote (puede ser <0)
    imbalance: float  # (bid-ask)/(bid+ask) del último quote, [-1,1]
    mid_drift: float  # log(mid_last/mid_first), 0.0 con 1 quote
    ofi: float  # order-flow imbalance normalizado por profundidad
    signed_volume_ratio: float  # (buy-sell)/(buy+sell), 0 sin trades
    buy_volume: float
    sell_volume: float
    trade_intensity: float  # trades/segundo sobre el span
    price_impact: float  # |mid_drift|/size operado (0 sin trades)
    crossed_pct: float  # fracción de quotes con bid > ask
    trade_price_vol: float  # std log precios operados (0 si <2 trades)


def _ofi_raw(quotes: tuple[Quote, ...]) -> tuple[float, float]:
    """(ofi_raw, depth_total) estilo Cont et al. simplificado a L1."""
    raw = 0.0
    depth = 0.0
    for prev, curr in zip(quotes, quotes[1:]):
        if curr.bid > prev.bid:
            raw += curr.bid_size
        elif curr.bid < prev.bid:
            raw -= prev.bid_size
        else:
            raw += curr.bid_size - prev.bid_size
        if curr.ask < prev.ask:
            raw -= curr.ask_size
        elif curr.ask > prev.ask:
            raw += prev.ask_size
        else:
            raw -= curr.ask_size - prev.ask_size
        depth += prev.bid_size + prev.ask_size + curr.bid_size + curr.ask_size
    return raw, depth


def l1_features(
    quotes: tuple[Quote, ...] | list[Quote],
    trades: tuple[Trade, ...] | list[Trade] = (),
) -> L1Features:
    """Extrae el vector L1 de un tramo de quotes (+ tape opcional)."""
    quotes = tuple(quotes)
    trades = tuple(trades)
    if len(quotes) < MIN_QUOTES:
        raise ValueError(
            f"quotes insuficientes para L1: {len(quotes)} < {MIN_QUOTES}"
        )
    symbol = quotes[0].symbol
    if any(q.symbol != symbol for q in quotes):
        raise ValueError("quotes mezclan símbolos")
    if any(t.symbol != symbol for t in trades):
        raise ValueError("trades de otro símbolo que los quotes")

    first, last = quotes[0], quotes[-1]
    span = max(last.timestamp - first.timestamp, 1e-9)
    mid_first, mid_last = first.midpoint, last.midpoint
    spread_bps = last.spread / mid_last * 10000.0
    depth_last = last.bid_size + last.ask_size
    imbalance = (
        (last.bid_size - last.ask_size) / depth_last if depth_last > 0 else 0.0
    )
    mid_drift = math.log(mid_last / mid_first) if mid_first > 0 else 0.0

    raw, depth = _ofi_raw(quotes)
    ofi = raw / depth if depth > 0 else 0.0

    buy = sum(t.size for t in trades if t.taker_side == "buy")
    sell = sum(t.size for t in trades if t.taker_side == "sell")
    signed = (buy - sell) / (buy + sell) if (buy + sell) > 0 else 0.0
    intensity = len(trades) / span
    impact = abs(mid_drift) / (buy + sell) if (buy + sell) > 0 else 0.0
    crossed = sum(1 for q in quotes if q.bid > q.ask) / len(quotes)

    prices = [t.price for t in trades]
    if len(prices) >= 2:
        logs = [math.log(p) for p in prices]
        mean = sum(logs) / len(logs)
        var = sum((x - mean) ** 2 for x in logs) / len(logs)
        price_vol = math.sqrt(var)
    else:
        price_vol = 0.0

    return L1Features(
        n_quotes=len(quotes),
        n_trades=len(trades),
        span_seconds=span,
        spread_bps=spread_bps,
        imbalance=imbalance,
        mid_drift=mid_drift,
        ofi=ofi,
        signed_volume_ratio=signed,
        buy_volume=buy,
        sell_volume=sell,
        trade_intensity=intensity,
        price_impact=impact,
        crossed_pct=crossed,
        trade_price_vol=price_vol,
    )
