"""Trading MVP 0.1 — Feed de klines cerradas de Binance (REST público).

Diseño:
- Endpoint PÚBLICO ``/api/v3/klines``: no requiere API key, no firma, no
  órdenes. Presupuesto del experimento: $0.
- Solo velas CERRADAS: una vela cuyo close_time aún no llegó no existe para
  el pipeline (la última fila del REST es la vela en formación y se ignora).
- Transporte inyectable: los tests pasan un ``http_get`` falso; producción
  usa requests con timeout. Nada de websockets ni dependencias nuevas.
- Gap tracking propio: si entre polls falta algún intervalo (caída de red,
  restart), se cuenta y queda visible en el status line. Nada se inventa.

El feed mantiene un buffer acotado de cierres recientes que sirve como
PriceLookup para el OutcomeResolver y como ventana de features.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Deque, Mapping

import requests

from iot_machine_learning.domain.entities.market import Candle, DataStatus

__all__ = ["BinanceKlinesFeed"]

_INTERVAL_TO_BINANCE: dict[int, str] = {
    60: "1m",
    180: "3m",
    300: "5m",
    900: "15m",
    1800: "30m",
    3600: "1h",
}

HttpGet = Callable[[str, Mapping[str, Any]], list[list[Any]]]


def _map_symbol(symbol: str) -> str:
    """BTC-USD → BTCUSDT (convención del resto del pipeline → Binance)."""
    mapped = symbol.replace("-", "").upper()
    if mapped.endswith("USD") and not mapped.endswith("USDT"):
        mapped = mapped[:-3] + "USDT"
    return mapped


def _binance_interval(interval_seconds: int) -> str:
    label = _INTERVAL_TO_BINANCE.get(interval_seconds)
    if label is None:
        raise ValueError(
            f"intervalo no soportado por el MVP: {interval_seconds}s "
            f"(soportados: {sorted(_INTERVAL_TO_BINANCE)})"
        )
    return label


def kline_array_to_candle(
    row: list[Any],
    *,
    symbol: str,
    interval_seconds: int,
) -> tuple[Candle, float]:
    """Convierte una fila REST kline a Candle cerrada.

    Retorna ``(candle, close_time_epoch)``; el llamador decide si la vela
    ya cerró. Estructura REST:
    [openTime, open, high, low, close, volume, closeTime, ...]
    """
    if len(row) < 7:
        raise ValueError(f"kline truncada: {len(row)} campos")
    open_time_ms = int(row[0])
    close_time_ms = int(row[6])
    candle = Candle(
        symbol=symbol,
        timestamp=open_time_ms / 1000.0,
        data_status=DataStatus.REALTIME,
        source_provider="binance-rest",
        venue=None,
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        interval_seconds=interval_seconds,
        vwap=None,
        trade_count=0,
        adjusted=False,
    )
    return candle, close_time_ms / 1000.0


class BinanceKlinesFeed:
    """Poller de velas cerradas BTC-USD (o símbolo equivalente).

    Args:
        symbol: símbolo del pipeline (BTC-USD); se mapea a BTCUSDT.
        interval_seconds: resolución de vela (60 ⇒ 1m).
        http_get: transporte inyectable ``(url, params) -> filas``.
            Default: requests GET contra ``base_url``.
        base_url: permite test/regiones (default https://api.binance.com).
        buffer_size: cierres recientes retenidos para features/price lookup.
    """

    def __init__(
        self,
        symbol: str = "BTC-USD",
        interval_seconds: int = 60,
        *,
        http_get: HttpGet | None = None,
        base_url: str = "https://api.binance.com",
        buffer_size: int = 600,
        timeout_seconds: float = 10.0,
    ) -> None:
        self.symbol = symbol
        self.interval_seconds = interval_seconds
        self._binance_symbol = _map_symbol(symbol)
        self._interval_label = _binance_interval(interval_seconds)
        self._http_get = http_get or self._requests_get
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds

        self._candles: Deque[Candle] = deque(maxlen=buffer_size)
        self._closes: Deque[tuple[float, float]] = deque(maxlen=buffer_size)
        self._last_open_ts: float | None = None
        # Estado observable por el runner (status line / uptime honesto):
        self.polls = 0
        self.errors = 0
        self.gaps = 0
        self.candles_received = 0
        self.last_success_monotonic: float | None = None
        self.last_error: str | None = None

    # ── contrato usado por el runner ────────────────────────────────────

    def poll_closed(self) -> tuple[Candle, ...]:
        """Trae velas cerradas nuevas desde el último poll.

        Nunca lanza por errores de red: registra el error, devuelve vacío y
        el ciclo siguiente reintenta desde el mismo punto (el servidor es
        la fuente de verdad de la continuidad).
        """
        params: dict[str, Any] = {
            "symbol": self._binance_symbol,
            "interval": self._interval_label,
            "limit": self._default_limit(),
        }
        if self._last_open_ts is not None:
            params["startTime"] = int((self._last_open_ts + 1) * 1000)
        self.polls += 1
        try:
            rows = self._http_get(f"{self._base_url}/api/v3/klines", params)
        except Exception as exc:  # noqa: BLE001 - el runner decide qué hacer
            self.errors += 1
            self.last_error = repr(exc)
            return ()
        self.last_error = None
        self.last_success_monotonic = time.monotonic()

        now_s = time.time()
        fresh: list[Candle] = []
        for row in rows:
            candle, close_time = kline_array_to_candle(
                row, symbol=self.symbol, interval_seconds=self.interval_seconds
            )
            if close_time > now_s:
                continue  # vela en formación: no existe para el pipeline
            if (
                self._last_open_ts is not None
                and candle.timestamp <= self._last_open_ts
            ):
                continue  # duplicado
            expected = (
                self._last_open_ts + self.interval_seconds
                if self._last_open_ts is not None
                else candle.timestamp
            )
            missing = round((candle.timestamp - expected) / self.interval_seconds)
            if missing > 0:
                self.gaps += missing
            fresh.append(candle)
            self._last_open_ts = candle.timestamp
            self.candles_received += 1
            self._candles.append(candle)
            self._closes.append((candle.timestamp, candle.close))
        return tuple(fresh)

    def recent_candles(self, limit: int | None = None) -> tuple[Candle, ...]:
        """Últimas ``limit`` velas cerradas (para la ventana de features)."""
        if limit is None or limit >= len(self._candles):
            return tuple(self._candles)
        return tuple(tuple(self._candles)[-limit:])

    def recent_closes(self) -> tuple[tuple[float, float], ...]:
        """Cierres recientes ``(timestamp, close)`` en orden temporal."""
        return tuple(self._closes)

    def last_close(self, at_or_before: float) -> float | None:
        """Contrato PriceLookup para el OutcomeResolver."""
        best: float | None = None
        for ts, close in self._closes:
            if ts <= at_or_before:
                best = close
            else:
                break
        return best

    @property
    def connected(self) -> bool:
        """True si hubo al menos un poll exitoso sin error posterior."""
        return (
            self.last_success_monotonic is not None
            and self.last_error is None
        )

    # ── internals ───────────────────────────────────────────────────────

    def _default_limit(self) -> int:
        # Suficiente para reconstruir ventana + resolver horizontes cortos
        # tras un restart, sin descargarse el histórico completo cada ciclo.
        return 300

    def _requests_get(self, url: str, params: Mapping[str, Any]) -> list[list[Any]]:
        response = requests.get(url, params=params, timeout=self._timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError(f"payload klines no es lista: {type(payload)!r}")
        return payload
