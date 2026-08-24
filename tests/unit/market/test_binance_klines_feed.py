"""Trading MVP 0.1 — BinanceKlinesFeed sin red (transporte inyectable)."""

from __future__ import annotations

import time

import pytest

from iot_machine_learning.infrastructure.adapters.market.binance_klines_feed import (
    BinanceKlinesFeed,
    _binance_interval,
    _map_symbol,
    kline_array_to_candle,
)


def _kline_row(open_ts_s: float, close: float = 100.0) -> list:
    open_ms = int(open_ts_s * 1000)
    return [
        open_ms,
        str(close - 1),
        str(close + 2),
        str(close - 3),
        str(close),
        "12.5",
        open_ms + 59_999,
        "0",
        "60",
        "1200",
        "0",
        "0",
    ]


class FakeTransport:
    """http_get falso: cola de respuestas (páginas o excepciones)."""

    def __init__(self, pages: list) -> None:
        self.pages = list(pages)
        self.calls: list[tuple[str, dict]] = []

    def __call__(self, url: str, params: dict) -> list[list]:
        self.calls.append((url, dict(params)))
        item = self.pages.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


# ─── Mapeos ─────────────────────────────────────────────────────────────────


class TestMappings:
    def test_symbol_pipeline_a_binance(self):
        assert _map_symbol("BTC-USD") == "BTCUSDT"
        assert _map_symbol("eth-usd") == "ETHUSDT"
        assert _map_symbol("BTCUSDT") == "BTCUSDT"

    def test_intervalos_soportados(self):
        assert _binance_interval(60) == "1m"
        assert _binance_interval(3600) == "1h"
        with pytest.raises(ValueError, match="no soportado"):
            _binance_interval(7)

    def test_kline_array_a_candle(self):
        candle, close_time = kline_array_to_candle(
            _kline_row(100.0), symbol="BTC-USD", interval_seconds=60
        )
        assert candle.symbol == "BTC-USD"
        assert candle.timestamp == 100.0
        assert candle.close == 100.0
        assert candle.interval_seconds == 60
        assert close_time == pytest.approx(159.999)  # open_ms + 59999


# ─── Poll ───────────────────────────────────────────────────────────────────


class TestPollClosed:
    def test_primera_poll_trae_cerradas_e_ignora_en_formacion(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 470.0)
        rows = [_kline_row(300.0, 101), _kline_row(360.0, 102), _kline_row(420.0)]
        transport = FakeTransport([rows])
        feed = BinanceKlinesFeed(
            interval_seconds=60, http_get=transport, base_url="http://x"
        )
        candles = feed.poll_closed()

        # Cierres: 359.999 y 419.999 < 470 ⇒ cerradas; 479.999 >= 470 ⇒
        # vela en formación, no existe para el pipeline.
        assert [c.timestamp for c in candles] == [300.0, 360.0]
        assert feed.candles_received == 2
        assert feed.polls == 1
        url, params = transport.calls[0]
        assert params["symbol"] == "BTCUSDT"
        assert params["interval"] == "1m"

    def test_segunda_poll_deduplica_y_avanza_starttime(self, monkeypatch):
        t = {"now": 600.0}  # 420 cierra en 479.99 y 480 en 539.99 ⇒ ambas
        monkeypatch.setattr(time, "time", lambda: t["now"])
        transport = FakeTransport([
            [_kline_row(420.0), _kline_row(480.0)],   # ciclo 1
            [],                                        # ciclo 2: nada nuevo
        ])
        feed = BinanceKlinesFeed(
            interval_seconds=60, http_get=transport, base_url="http://x"
        )
        first = feed.poll_closed()
        assert [c.close for c in first] == [100.0, 100.0]

        _, params2 = None, None
        feed.poll_closed()
        _, params2 = transport.calls[1]
        # startTime avanza más allá del último open visto (480s → 481000 ms).
        assert params2["startTime"] == 481_000
        assert feed.candles_received == 2  # no duplica

    def test_error_de_red_registra_y_devuelve_vacio(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 10**12)
        transport = FakeTransport([ConnectionError("timeout")])
        feed = BinanceKlinesFeed(
            interval_seconds=60, http_get=transport, base_url="http://x"
        )
        assert feed.poll_closed() == ()
        assert feed.errors == 1
        assert feed.last_error is not None
        assert not feed.connected

    def test_gap_entre_velas_se_cuenta_sin_inventar(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 10**6)
        transport = FakeTransport([
            [_kline_row(60.0)],
            [_kline_row(600.0)],  # salto: faltan 60..540 ⇒ 9 velas
        ])
        feed = BinanceKlinesFeed(
            interval_seconds=60, http_get=transport, base_url="http://x"
        )
        feed.poll_closed()
        feed.poll_closed()
        assert feed.gaps == 8
        assert feed.last_close(700.0) == 100.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
