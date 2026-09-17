"""FASE 0 — Auditoría extrema de ingesta.

Cada tick/candle/quote que entra debe responder: ¿correcto, completo,
sincronizado, duplicado, retrasado, perdido o mal transformado?
Nada se descarta en silencio: todo se cuenta y se expone.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.entities.market.audit import (
    AuditedEvent,
    audit_feed,
)
from iot_machine_learning.domain.entities.market import (
    Candle,
    DataStatus,
    Quote,
    Trade,
)


def _ev(ts: float, **kw) -> AuditedEvent:
    base = {"timestamp": ts, "symbol": "BTC-USD", "kind": "candle", "status": "realtime"}
    base.update(kw)
    return AuditedEvent(**base)


class TestDuplicates:
    def test_ts_repetido_se_cuenta_no_se_descarta(self):
        report = audit_feed(
            [_ev(60.0), _ev(120.0), _ev(120.0)],
            symbol="BTC-USD", expected_interval_seconds=60,
        )
        assert report.total == 3
        assert report.duplicates == 1
        assert report.unique == 2
        assert any(a.kind == "duplicate" for a in report.anomalies)


class TestGaps:
    def test_salto_60_a_600_pierde_8_intervalos(self):
        events = [_ev(60.0), _ev(600.0)]
        report = audit_feed(events, symbol="BTC-USD", expected_interval_seconds=60)
        assert report.missing_intervals == 8
        assert len(report.gap_events) == 1
        # 2 únicos de 10 esperados → 20%
        assert report.completeness_pct == pytest.approx(20.0)

    def test_tramo_sano_100_porciento(self):
        events = [_ev(float(t)) for t in (0.0, 60.0, 120.0, 180.0)]
        report = audit_feed(events, symbol="BTC-USD", expected_interval_seconds=60)
        assert report.missing_intervals == 0
        assert report.completeness_pct == pytest.approx(100.0)


class TestOutOfOrder:
    def test_llegada_tardia_se_marca(self):
        report = audit_feed(
            [_ev(120.0), _ev(60.0)], symbol="BTC-USD",
            expected_interval_seconds=60,
        )
        assert report.out_of_order == 1
        assert any(a.kind == "out_of_order" for a in report.anomalies)


class TestStale:
    def test_lag_mayor_al_umbral_es_stale(self):
        events = [
            _ev(100.0, arrival_ts=100.4),
            _ev(160.0, arrival_ts=175.0),  # lag 15s
        ]
        report = audit_feed(
            events, symbol="BTC-USD", expected_interval_seconds=60,
            stale_lag_seconds=5.0,
        )
        assert report.stale_count == 1
        assert report.max_lag_seconds == pytest.approx(15.0)
        assert report.avg_lag_seconds == pytest.approx((0.4 + 15.0) / 2)


class TestMixing:
    def test_symbol_mix_visible(self):
        report = audit_feed(
            [_ev(60.0), _ev(120.0, symbol="SPY")], symbol="BTC-USD",
            expected_interval_seconds=60,
        )
        assert "SPY" in report.symbols_seen
        assert any(a.kind == "symbol_mix" for a in report.anomalies)

    def test_replay_mezclado_con_live_se_marca(self):
        report = audit_feed(
            [_ev(60.0, status="replay"), _ev(120.0, status="realtime")],
            symbol="BTC-USD", expected_interval_seconds=60,
        )
        assert report.extra["replay_mixed_with_live"] is True


class TestSequence:
    def test_salto_y_rewind_de_seq(self):
        events = [
            _ev(60.0, seq=10), _ev(120.0, seq=13),  # faltan 11,12
            _ev(180.0, seq=11),  # rewind
            _ev(240.0, seq=13),  # duplicada
        ]
        report = audit_feed(events, symbol="BTC-USD", expected_interval_seconds=60)
        assert report.seq_gaps == 2
        assert report.seq_rewinds == 1
        assert report.seq_duplicates == 1


class TestDomainTransforms:
    """El dominio rechaza transformaciones incorrectas en la frontera."""

    def test_candle_con_low_mayor_que_open_falla(self):
        with pytest.raises(ValueError, match="low > min"):
            Candle(
                symbol="BTC-USD", timestamp=60.0,
                data_status=DataStatus.REALTIME, source_provider="binance",
                open=100.0, high=101.0, low=100.5, close=99.0,
                volume=1.0, interval_seconds=60,
            )

    def test_trade_size_cero_falla(self):
        with pytest.raises(ValueError, match="size no puede ser 0"):
            Trade(
                symbol="BTC-USD", timestamp=60.0,
                data_status=DataStatus.REALTIME, source_provider="binance",
                price=100.0, size=0.0,
            )

    def test_quote_acepta_libro_cruzado_pero_no_basura(self):
        q = Quote(
            symbol="SPY", timestamp=60.0, data_status=DataStatus.DELAYED,
            source_provider="alpaca", bid=100.0, bid_size=1.0,
            ask=99.9, ask_size=1.0,  # cruzado: real, se acepta
        )
        assert q.spread < 0
        with pytest.raises(ValueError, match="bid"):
            Quote(
                symbol="SPY", timestamp=60.0,
                data_status=DataStatus.DELAYED, source_provider="alpaca",
                bid=float("nan"), bid_size=1.0, ask=100.0, ask_size=1.0,
            )


class TestWsFeedMapping:
    """Los factories WS deben emitir DataStatus válidos (bug LIVE/CLOSED)."""

    def test_trade_quote_book_y_candle_usan_realtime_o_unverified(self):
        from iot_machine_learning.infrastructure.adapters.market.binance.ws_feed import (
            BinanceWSFeed,
        )
        from types import SimpleNamespace

        feed = BinanceWSFeed.__new__(BinanceWSFeed)
        feed._ws_client = SimpleNamespace(symbol="BTCUSDT")
        trade = feed._create_trade(
            {"T": 60_000, "p": "100", "q": "0.5", "a": 1, "m": False}, 61.0
        )
        assert trade.data_status is DataStatus.REALTIME
        assert trade.is_live

        quote = feed._create_quote(
            {"b": "99", "B": "1", "a": "101", "A": "2"}, 62.0
        )
        assert quote.data_status is DataStatus.REALTIME

        forming = feed._create_candle(
            {"k": {"t": 60_000, "x": False, "o": "1", "h": "2",
                   "l": "0.5", "c": "1.5", "v": "10", "i": "1m"}}, 63.0
        )
        closed = feed._create_candle(
            {"k": {"t": 60_000, "x": True, "o": "1", "h": "2",
                   "l": "0.5", "c": "1.5", "v": "10", "i": "1m"}}, 63.0
        )
        assert forming.data_status is DataStatus.UNVERIFIED
        assert not forming.is_live  # en formación NO es señal
        assert closed.data_status is DataStatus.REALTIME
        assert closed.is_live


class TestKlinesCounters:
    def test_duplicado_y_tardio_se_cuentan(self, monkeypatch):
        import time

        from iot_machine_learning.infrastructure.adapters.market.feeds.binance_klines_feed import (
            BinanceKlinesFeed,
        )

        def _row(open_ts: float):
            ms = int(open_ts * 1000)
            return [ms, "99", "102", "97", "100", "1", ms + 59_999,
                    "0", "60", "1", "0", "0"]

        monkeypatch.setattr(time, "time", lambda: 10**6)
        pages = [[_row(60.0)], [_row(60.0)], [_row(30.0)]]
        feed = BinanceKlinesFeed(
            interval_seconds=60,
            http_get=lambda url, params: pages.pop(0),
            base_url="http://x",
        )
        feed.poll_closed()  # 60.0 nuevo
        feed.poll_closed()  # 60.0 duplicado
        feed.poll_closed()  # 30.0 tardío
        assert feed.duplicates_skipped == 1
        assert feed.late_skipped == 1
        assert feed.candles_received == 1
