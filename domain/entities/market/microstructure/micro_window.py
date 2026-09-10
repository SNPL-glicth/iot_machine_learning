"""Ventana microestructural (FASE 3).

Buffer inmutable (append devuelve ventana nueva) de Quotes+Trades del
mismo símbolo, ordenados por timestamp. Puentes:
- ``features()`` → vector L1 puro;
- ``audited()`` → eventos para el auditor de Fase 0 (dups, gaps,
  out-of-order, symbol-mix sobre el tape);
- ``l1_available(profile)`` → gate por Capability (TRADES+QUOTES).
"""

from __future__ import annotations

from ..audit import AuditedEvent
from ..capability import Capability, ProviderProfile
from ..observations import Quote, Trade
from .l1_features import L1Features, l1_features

__all__ = ["MicroWindow", "l1_available"]


def l1_available(profile: ProviderProfile) -> bool:
    """True si el provider entrega tape L1 (trades + quotes)."""
    if not isinstance(profile, ProviderProfile):
        raise TypeError("profile debe ser ProviderProfile")
    caps = profile.capabilities
    return Capability.TRADES in caps and Capability.QUOTES in caps


class MicroWindow:
    """Ventana append-only de microestructura (inmutable)."""

    __slots__ = ("symbol", "_quotes", "_trades")

    def __init__(
        self,
        symbol: str,
        quotes: tuple[Quote, ...] = (),
        trades: tuple[Trade, ...] = (),
    ) -> None:
        symbol = symbol.strip()
        if not symbol:
            raise ValueError("symbol no puede ser vacío")
        for q in quotes:
            if not isinstance(q, Quote):
                raise TypeError("quotes debe contener Quote")
            if q.symbol != symbol:
                raise ValueError(f"quote de otro símbolo: {q.symbol!r}")
        for t in trades:
            if not isinstance(t, Trade):
                raise TypeError("trades debe contener Trade")
            if t.symbol != symbol:
                raise ValueError(f"trade de otro símbolo: {t.symbol!r}")
        ordered_q = tuple(sorted(quotes, key=lambda q: q.timestamp))
        ordered_t = tuple(sorted(trades, key=lambda t: t.timestamp))
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "_quotes", ordered_q)
        object.__setattr__(self, "_trades", ordered_t)

    @property
    def quotes(self) -> tuple[Quote, ...]:
        return self._quotes

    @property
    def trades(self) -> tuple[Trade, ...]:
        return self._trades

    @property
    def size(self) -> int:
        return len(self._quotes) + len(self._trades)

    def append_quote(self, quote: Quote) -> MicroWindow:
        """Agrega un quote (debe ser más nuevo; si no, ValueError)."""
        if not isinstance(quote, Quote):
            raise TypeError("esperaba Quote")
        if quote.symbol != self.symbol:
            raise ValueError(f"quote de otro símbolo: {quote.symbol!r}")
        if self._quotes and quote.timestamp <= self._quotes[-1].timestamp:
            raise ValueError("quote fuera de orden o duplicado")
        return MicroWindow(self.symbol, self._quotes + (quote,), self._trades)

    def append_trade(self, trade: Trade) -> MicroWindow:
        """Agrega un trade (debe ser más nuevo; si no, ValueError)."""
        if not isinstance(trade, Trade):
            raise TypeError("esperaba Trade")
        if trade.symbol != self.symbol:
            raise ValueError(f"trade de otro símbolo: {trade.symbol!r}")
        if self._trades and trade.timestamp <= self._trades[-1].timestamp:
            raise ValueError("trade fuera de orden o duplicado")
        return MicroWindow(self.symbol, self._quotes, self._trades + (trade,))

    def features(self) -> L1Features:
        """Vector L1 del tramo (requiere ≥2 quotes)."""
        return l1_features(self._quotes, self._trades)

    def audited(self) -> tuple[AuditedEvent, ...]:
        """Eventos en orden temporal para ``audit_feed`` (Fase 0).

        Sin arrival_ts (replay): detecta duplicados, gaps y mezcla de
        símbolos; el orden de llegada lo audita el adapter live.
        """
        events = [
            AuditedEvent(
                timestamp=q.timestamp, symbol=q.symbol, kind="quote",
                status=q.data_status.value,
            )
            for q in self._quotes
        ]
        events += [
            AuditedEvent(
                timestamp=t.timestamp, symbol=t.symbol, kind="trade",
                status=t.data_status.value,
            )
            for t in self._trades
        ]
        return tuple(sorted(events, key=lambda e: e.timestamp))
