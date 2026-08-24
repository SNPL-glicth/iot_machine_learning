"""Binance candle parser."""

from __future__ import annotations

from typing import Any, Mapping

from iot_machine_learning.domain.entities.market import Candle, DataStatus
from .helpers import require, as_float, ms_to_epoch
from .constants import _KLINE_INTERVALS


def candle_from_payload(
    payload: Mapping[str, object],
    *,
    interval_seconds: int | None = None,
    received_at: float | None = None,
    provider_name: str = "binance",
    profile_has_realtime: bool = True,
) -> Candle:
    if payload.get("e") != "kline":
        raise ValueError(
            f"esperaba evento 'kline' de Binance, "
            f"obtenido {payload.get('e')!r}"
        )
    kline = require(payload, "k", label="k (kline)")
    if not isinstance(kline, Mapping):
        raise TypeError(f"'k' de Binance debe ser un objeto: {kline!r}")
    interval = None
    if interval_seconds is None:
        raw_interval = kline.get("i")
        if isinstance(raw_interval, str):
            interval = _KLINE_INTERVALS.get(raw_interval)
        if interval is None:
            raise ValueError(
                "interval_seconds requerido: intervalo de kline "
                f"{raw_interval!r} no reconocido"
            )
    else:
        interval = interval_seconds
    symbol = str(require(payload, "s"))
    timestamp = ms_to_epoch(require(kline, "t", label="k.t"), "k.t")
    data_status = DataStatus.REALTIME if profile_has_realtime else DataStatus.DELAYED
    return Candle(
        symbol=symbol,
        timestamp=timestamp,
        data_status=data_status,
        source_provider=provider_name,
        venue=None,
        open=as_float(require(kline, "o", label="k.o"), "k.o"),
        high=as_float(require(kline, "h", label="k.h"), "k.h"),
        low=as_float(require(kline, "l", label="k.l"), "k.l"),
        close=as_float(require(kline, "c", label="k.c"), "k.c"),
        volume=as_float(require(kline, "v", label="k.v"), "k.v"),
        interval_seconds=interval,
        vwap=None,
        trade_count=int(require(kline, "n", label="k.n")),
        adjusted=False,
    )