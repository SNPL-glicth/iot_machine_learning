"""Tests unitarios — entidades de dominio ZENIN Market (FASE 2).

Contrato v1: tipado fuerte, validación y inmutabilidad.
Sin depender de infraestructura.
"""

from __future__ import annotations

import dataclasses

import pytest
from iot_machine_learning.domain.entities.market import (
    Candle,
    Capability,
    DataStatus,
    MarketObservation,
    OrderBookSnapshot,
    ProviderProfile,
    Quote,
    Trade,
)


def _with_defaults(**defaults):
    def build(**overrides) -> object:
        merged = dict(defaults)
        merged.update(overrides)
        return merged

    return build


_trade_kwargs = _with_defaults(
    symbol="NVDA",
    timestamp=1_600_000_000.123,
    data_status=DataStatus.REALTIME,
    source_provider="alpaca",
    price=181.42,
    size=100.0,
)


def _trade(**overrides) -> Trade:
    return Trade(**_trade_kwargs(**overrides))


_quote_kwargs = _with_defaults(
    symbol="NVDA",
    timestamp=1_600_000_000.123,
    data_status=DataStatus.REALTIME,
    source_provider="alpaca",
    bid=181.40,
    bid_size=5.0,
    ask=181.43,
    ask_size=7.0,
)


def _quote(**overrides) -> Quote:
    return Quote(**_quote_kwargs(**overrides))


_candle_kwargs = _with_defaults(
    symbol="NVDA",
    timestamp=1_600_000_000.0,
    data_status=DataStatus.REALTIME,
    source_provider="alpaca",
    open=180.0,
    high=182.0,
    low=179.5,
    close=181.5,
    volume=1_000_000.0,
    interval_seconds=60,
)


def _candle(**overrides) -> Candle:
    return Candle(**_candle_kwargs(**overrides))


class TestProviderProfile:
    def test_valid(self):
        p = ProviderProfile(
            provider="alpaca",
            asset_class="equities",
            capabilities=frozenset(
                {Capability.TRADES, Capability.QUOTES, Capability.REALTIME}
            ),
            max_ws_symbols=30,
        )
        assert p.provider == "alpaca"
        assert p.has_realtime
        assert not p.has_order_book

    def test_empty_provider_rejected(self):
        with pytest.raises(ValueError):
            ProviderProfile(provider="  ", asset_class="equities")

    def test_invalid_capability_rejected(self):
        with pytest.raises(TypeError):
            ProviderProfile(
                provider="x",
                asset_class="y",
                capabilities=frozenset({"NOT_A_CAPABILITY"}),  # type: ignore[arg-type]
            )

    def test_invalid_max_ws_rejected(self):
        with pytest.raises(ValueError):
            ProviderProfile(provider="x", asset_class="y", max_ws_symbols=0)

    def test_immutable(self):
        p = ProviderProfile(provider="x", asset_class="y")
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.provider = "z"  # type: ignore[misc]


class TestDataStatus:
    def test_live_signal(self):
        assert DataStatus.REALTIME.is_live_signal
        assert not DataStatus.DELAYED.is_live_signal
        assert not DataStatus.REPLAY.is_live_signal

    def test_invalid_status_rejected(self):
        with pytest.raises(TypeError):
            Trade(
                symbol="NVDA",
                timestamp=1.0,
                data_status="realtime",  # type: ignore[arg-type]
                source_provider="alpaca",
                price=1.0,
                size=1.0,
            )


class TestTrade:
    def test_valid(self):
        t = _trade(taker_side="buy", conditions=("T",), tape="C")
        assert t.is_live
        assert t.conditions == ("T",)
        assert t.venue is None

    def test_zero_size_rejected(self):
        with pytest.raises(ValueError):
            _trade(size=0.0)

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError):
            _trade(price=-1.0)

    def test_nan_price_rejected(self):
        with pytest.raises(ValueError):
            _trade(price=float("nan"))

    def test_bad_taker_side_rejected(self):
        with pytest.raises(ValueError):
            _trade(taker_side="maybe")

    def test_bad_symbol_rejected(self):
        with pytest.raises(ValueError):
            _trade(symbol="NV DA")

    def test_empty_symbol_rejected(self):
        with pytest.raises(ValueError):
            _trade(symbol="   ")

    def test_invalid_timestamp_rejected(self):
        with pytest.raises(ValueError):
            _trade(timestamp=0.0)

    def test_immutable(self):
        t = _trade()
        with pytest.raises(dataclasses.FrozenInstanceError):
            t.price = 999999.0  # type: ignore[misc]


class TestQuote:
    def test_valid(self):
        q = _quote(bid_exchange="N", ask_exchange="Q")
        assert q.spread == pytest.approx(0.03)
        assert q.midpoint == pytest.approx(181.415)

    def test_crossed_book_allowed(self):
        """Libros cruzados existen en feeds reales: no se rechazan."""
        q = _quote(bid=182.0, ask=181.0)
        assert q.spread == pytest.approx(-1.0)

    def test_negative_bid_rejected(self):
        with pytest.raises(ValueError):
            _quote(bid=-1.0)

    def test_negative_size_rejected(self):
        with pytest.raises(ValueError):
            _quote(bid_size=-2.0)


class TestCandle:
    def test_valid(self):
        c = _candle(vwap=180.9, trade_count=120)
        assert c.is_bullish
        assert c.body == pytest.approx(1.5)

    def test_low_above_open_rejected(self):
        with pytest.raises(ValueError):
            _candle(low=181.0, open=180.0)

    def test_high_below_close_rejected(self):
        with pytest.raises(ValueError):
            _candle(high=181.0, close=182.0)

    def test_zero_interval_rejected(self):
        with pytest.raises(ValueError):
            _candle(interval_seconds=0)

    def test_bearish(self):
        c = _candle(open=182.0, high=183.0, low=179.0, close=180.0)
        assert not c.is_bullish

    def test_immutable(self):
        c = _candle()
        with pytest.raises(dataclasses.FrozenInstanceError):
            c.close = 999999.0  # type: ignore[misc]


class TestOrderBookSnapshot:
    def test_valid(self):
        book = OrderBookSnapshot(
            symbol="BTC/USD",
            timestamp=1_600_000_000.0,
            data_status=DataStatus.REALTIME,
            source_provider="binance",
            bids=((100.0, 1.0), (99.5, 2.0)),
            asks=((100.5, 3.0), (101.0, 1.0)),
        )
        assert book.best_bid == pytest.approx(100.0)
        assert book.best_ask == pytest.approx(100.5)

    def test_unsorted_bids_rejected(self):
        with pytest.raises(ValueError):
            OrderBookSnapshot(
                symbol="BTC/USD",
                timestamp=1_600_000_000.0,
                data_status=DataStatus.REALTIME,
                source_provider="binance",
                bids=((99.5, 1.0), (100.0, 2.0)),
                asks=((100.5, 1.0),),
            )

    def test_empty_book_rejected(self):
        with pytest.raises(ValueError):
            OrderBookSnapshot(
                symbol="BTC/USD",
                timestamp=1_600_000_000.0,
                data_status=DataStatus.REALTIME,
                source_provider="binance",
                bids=(),
                asks=(),
            )

    def test_imbalance(self):
        book = OrderBookSnapshot(
            symbol="BTC/USD",
            timestamp=1_600_000_000.0,
            data_status=DataStatus.REALTIME,
            source_provider="binance",
            bids=((100.0, 3.0),),
            asks=((100.5, 1.0),),
        )
        assert book.imbalance == pytest.approx(0.5)


class TestMarketObservationBase:
    def test_empty_source_provider_rejected(self):
        with pytest.raises(ValueError):
            MarketObservation(
                symbol="NVDA",
                timestamp=1.0,
                data_status=DataStatus.REALTIME,
                source_provider="  ",
            )
