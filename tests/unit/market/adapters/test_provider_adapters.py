"""FASE 4 — contract tests de adapters de proveedores de mercado.

El mismo conjunto de aserciones corre contra AlpacaAdapter y
BinanceAdapter. Si un provider no declara la ``Capability`` necesaria,
el contrato se omite (``pytest.skip``) en lugar de fallar: así el
contrato se mantiene honesto respecto al perfil declarado.

Los mapeos específicos de cada proveedor (conditions, tape, taker_side,
intervalos, errores) viven en las clases ``TestAlpacaSpecific`` y
``TestBinanceSpecific``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import cast

import pytest
from iot_machine_learning.domain.entities.market import (
    Candle,
    Capability,
    DataStatus,
    OrderBookSnapshot,
    Quote,
    Trade,
)
from iot_machine_learning.domain.ports.market_data_provider import MarketDataProvider
from iot_machine_learning.infrastructure.adapters.market import (
    AlpacaAdapter,
    BinanceAdapter,
)

ADAPTERS: list[MarketDataProvider] = [AlpacaAdapter(), BinanceAdapter()]


def _iso_epoch(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


@pytest.fixture(params=ADAPTERS)
def adapter(request: pytest.FixtureRequest) -> MarketDataProvider:
    return cast(MarketDataProvider, request.param)


@pytest.fixture
def trade_payload(
    adapter: MarketDataProvider,
    load_payload: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    name = "alpaca_trade.json" if adapter.provider_name == "alpaca" else "binance_aggtrade.json"
    return load_payload(name)


@pytest.fixture
def quote_payload(
    adapter: MarketDataProvider,
    load_payload: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    name = "alpaca_quote.json" if adapter.provider_name == "alpaca" else "binance_bookticker.json"
    return load_payload(name)


@pytest.fixture
def candle_payload(
    adapter: MarketDataProvider,
    load_payload: Callable[[str], dict[str, object]],
) -> dict[str, object]:
    name = "alpaca_bar.json" if adapter.provider_name == "alpaca" else "binance_kline.json"
    return load_payload(name)


class TestAdapterContract:
    """Mismo conjunto de reglas para todos los proveedores."""

    def test_profile_realtime(self, adapter: MarketDataProvider) -> None:
        assert Capability.REALTIME in adapter.profile.capabilities
        assert adapter.profile.max_ws_symbols is None

    def test_trade_fields(
        self,
        adapter: MarketDataProvider,
        trade_payload: dict[str, object],
    ) -> None:
        trade = adapter.trade_from_payload(trade_payload)
        assert isinstance(trade, Trade)
        assert trade.source_provider == adapter.provider_name
        assert trade.data_status == DataStatus.REALTIME
        assert trade.price > 0
        assert trade.size > 0
        assert isinstance(trade.trade_id, str) and trade.trade_id
        assert isinstance(trade.timestamp, float) and trade.timestamp > 0

    def test_quote_fields(
        self,
        adapter: MarketDataProvider,
        quote_payload: dict[str, object],
    ) -> None:
        quote = adapter.quote_from_payload(quote_payload, received_at=1_700_000_000.0)
        assert isinstance(quote, Quote)
        assert quote.source_provider == adapter.provider_name
        assert quote.bid > 0 and quote.ask > 0
        assert quote.bid_size > 0 and quote.ask_size > 0
        assert isinstance(quote.timestamp, float)

    def test_candle_fields(
        self,
        adapter: MarketDataProvider,
        candle_payload: dict[str, object],
    ) -> None:
        candle = adapter.candle_from_payload(
            candle_payload,
            interval_seconds=900,
            received_at=1_700_000_000.0,
        )
        assert isinstance(candle, Candle)
        assert candle.source_provider == adapter.provider_name
        assert candle.open > 0 and candle.close > 0
        assert candle.high >= max(candle.open, candle.close)
        assert candle.low <= min(candle.open, candle.close)
        assert candle.volume > 0
        assert candle.interval_seconds > 0

    def test_order_book_contract(
        self,
        adapter: MarketDataProvider,
        load_payload: Callable[[str], dict[str, object]],
    ) -> None:
        if not adapter.profile.has_order_book:
            pytest.skip("provider no declara ORDER_BOOK_L2")
        depth = load_payload("binance_depth.json")
        book = adapter.order_book_from_payload(
            depth, symbol="BTCUSDT", received_at=1_700_000_000.0
        )
        assert book.reset is True
        assert book.bids[0][0] >= book.bids[-1][0]
        assert book.asks[0][0] <= book.asks[-1][0]

    def test_parse_dispatch_matches_specific_method(
        self,
        adapter: MarketDataProvider,
        trade_payload: dict[str, object],
    ) -> None:
        parsed = adapter.parse(trade_payload)
        direct = adapter.trade_from_payload(trade_payload)
        assert parsed == direct

    def test_parse_rejects_foreign_payload(
        self,
        adapter: MarketDataProvider,
        load_payload: Callable[[str], dict[str, object]],
    ) -> None:
        foreign = (
            "binance_aggtrade.json"
            if adapter.provider_name == "alpaca"
            else "alpaca_trade.json"
        )
        with pytest.raises(ValueError):
            adapter.parse(load_payload(foreign))

    def test_parse_rejects_unknown_event(self, adapter: MarketDataProvider) -> None:
        with pytest.raises(ValueError):
            adapter.parse({"T": "zz", "e": "zz"})


class TestAlpacaSpecific:
    """Mapeo particular de Alpaca (actions, tape, ISO-8601 con Z)."""

    def test_trade_mapping(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        payload = load_payload("alpaca_trade.json")
        trade = AlpacaAdapter().trade_from_payload(payload)
        assert trade.symbol == "AAPL"
        assert trade.price == 226.5
        assert trade.size == 100
        assert trade.trade_id == "2855474380"
        assert trade.venue == "V"
        assert trade.conditions == ("T",)
        assert trade.tape == "C"
        assert trade.corrected is True
        assert trade.taker_side is None
        assert trade.timestamp == _iso_epoch("2023-03-23T14:36:34.9204535Z")

    def test_quote_mapping(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        payload = load_payload("alpaca_quote.json")
        quote = AlpacaAdapter().quote_from_payload(payload)
        assert quote.bid == 226.49
        assert quote.bid_size == 1
        assert quote.ask == 226.5
        assert quote.ask_size == 3
        assert quote.bid_exchange == "V"
        assert quote.ask_exchange == "Q"
        assert quote.conditions == ("R",)
        assert quote.tape == "C"

    def test_bar_mapping(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        payload = load_payload("alpaca_bar.json")
        candle = AlpacaAdapter().candle_from_payload(payload, interval_seconds=900)
        assert candle.open == 226.49
        assert candle.high == 226.5
        assert candle.low == 226.48
        assert candle.close == 226.5
        assert candle.volume == 100
        assert candle.vwap == 226.495
        assert candle.trade_count == 5
        assert candle.interval_seconds == 900
        assert candle.adjusted is False
        assert candle.timestamp == _iso_epoch("2023-03-23T14:36:34Z")

    def test_bar_requires_interval(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        payload = load_payload("alpaca_bar.json")
        with pytest.raises(ValueError, match="interval_seconds"):
            AlpacaAdapter().candle_from_payload(payload)

    def test_parse_dispatch_by_event_type(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        adapter = AlpacaAdapter()
        assert isinstance(adapter.parse(load_payload("alpaca_trade.json")), Trade)
        assert isinstance(adapter.parse(load_payload("alpaca_quote.json")), Quote)
        assert isinstance(
            adapter.parse(load_payload("alpaca_bar.json"), interval_seconds=60),
            Candle,
        )

    def test_no_order_book(self) -> None:
        with pytest.raises(NotImplementedError):
            AlpacaAdapter().order_book_from_payload({})


class TestBinanceSpecific:
    """Mapeo particular de Binance (epoch ms, strings, bookTicker)."""

    def test_agg_trade_mapping(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        payload = load_payload("binance_aggtrade.json")
        trade = BinanceAdapter().trade_from_payload(payload)
        assert trade.symbol == "BTCUSDT"
        assert trade.price == 0.001
        assert trade.size == 100
        assert trade.trade_id == "5933014"
        assert trade.taker_side == "sell"
        assert trade.venue is None
        assert trade.conditions == ()
        assert trade.corrected is False
        assert trade.timestamp == 123456.785

    def test_agg_trade_maker_false_is_buy(self) -> None:
        payload: dict[str, object] = {
            "e": "aggTrade",
            "s": "BTCUSDT",
            "a": 1,
            "p": "0.001",
            "q": "100",
            "T": 123456785,
            "m": False,
            "M": True,
        }
        trade = BinanceAdapter().trade_from_payload(payload)
        assert trade.taker_side == "buy"

    def test_book_ticker_uses_received_at(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        payload = load_payload("binance_bookticker.json")
        quote = BinanceAdapter().quote_from_payload(payload, received_at=123.5)
        assert quote.symbol == "BNBUSDT"
        assert quote.bid == 25.3519
        assert quote.bid_size == 31.21
        assert quote.ask == 25.3652
        assert quote.ask_size == 40.66
        assert quote.timestamp == 123.5
        assert quote.venue is None
        assert quote.spread == pytest.approx(0.0133)

    def test_book_ticker_requires_received_at(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        payload = load_payload("binance_bookticker.json")
        with pytest.raises(ValueError, match="received_at"):
            BinanceAdapter().quote_from_payload(payload)

    def test_kline_derives_interval(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        payload = load_payload("binance_kline.json")
        candle = BinanceAdapter().candle_from_payload(payload)
        assert candle.symbol == "BTCUSDT"
        assert candle.interval_seconds == 60
        assert candle.open == 0.001
        assert candle.close == 0.002
        assert candle.high == 0.0025
        assert candle.low == 0.0009
        assert candle.volume == 1000
        assert candle.trade_count == 100
        assert candle.vwap is None
        assert candle.timestamp == 123400.0

    def test_kline_unknown_interval_requires_parameter(self) -> None:
        payload: dict[str, object] = {
            "e": "kline",
            "s": "BTCUSDT",
            "k": {
                "t": 123400000,
                "o": "1",
                "c": "2",
                "h": "2",
                "l": "1",
                "v": "1",
                "n": 1,
                "i": "7h",
            },
        }
        with pytest.raises(ValueError, match="interval_seconds"):
            BinanceAdapter().candle_from_payload(payload)
        candle = BinanceAdapter().candle_from_payload(payload, interval_seconds=14400)
        assert candle.interval_seconds == 14400

    def test_depth_snapshot_mapping(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        payload = load_payload("binance_depth.json")
        book = BinanceAdapter().order_book_from_payload(
            payload, symbol="BTCUSDT", received_at=100.0
        )
        assert book.symbol == "BTCUSDT"
        assert book.best_bid == 4.0
        assert book.best_ask == 4.000002
        assert book.bids[0] == (4.0, 431.0)
        assert book.asks[0] == (4.000002, 12.0)
        assert book.reset is True
        assert book.timestamp == 100.0

    def test_depth_requires_symbol(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        payload = load_payload("binance_depth.json")
        with pytest.raises(ValueError, match="symbol"):
            BinanceAdapter().order_book_from_payload(payload, received_at=100.0)

    def test_depth_requires_received_at(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        payload = load_payload("binance_depth.json")
        with pytest.raises(ValueError, match="received_at"):
            BinanceAdapter().order_book_from_payload(payload, symbol="BTCUSDT")

    def test_parse_dispatch(
        self, load_payload: Callable[[str], dict[str, object]]
    ) -> None:
        adapter = BinanceAdapter()
        assert isinstance(adapter.parse(load_payload("binance_aggtrade.json")), Trade)
        assert isinstance(adapter.parse(load_payload("binance_kline.json")), Candle)
        assert isinstance(
            adapter.parse(load_payload("binance_bookticker.json"), received_at=1.0),
            Quote,
        )
        assert isinstance(
            adapter.parse(
                load_payload("binance_depth.json"),
                symbol="BTCUSDT",
                received_at=1.0,
            ),
            OrderBookSnapshot,
        )
