"""Alpaca candle parser."""

from __future__ import annotations

from typing import Any, Mapping

from iot_machine_learning.domain.entities.market import Candle, DataStatus
from .helpers import require, as_float, iso_to_epoch


def candle_from_payload(
    payload: Mapping[str, object],
    *,
    interval_seconds: int | None = None,
    received_at: float | None = None,
    provider_name: str = "alpaca",
    profile_has_realtime: bool = True,
) -> Candle:
    if payload.get("T") != "b":
        raise ValueError(
            f"esperaba evento 'b' de Alpaca, obtenido {payload.get('T')!r}"
        )
    if interval_seconds is None:
        raise ValueError(
            "interval_seconds es requerido: el stream de bars de Alpaca "
            "no incluye el intervalo en el payload"
        )
    symbol = str(require(payload, "S"))
    timestamp = iso_to_epoch(str(require(payload, "t")))
    vwap_raw = payload.get("vw")
    data_status = DataStatus.REALTIME if profile_has_realtime else DataStatus.DELAYED
    return Candle(
        symbol=symbol,
        timestamp=timestamp,
        data_status=data_status,
        source_provider=provider_name,
        venue=None,
        open=as_float(require(payload, "o"), "o"),
        high=as_float(require(payload, "h"), "h"),
        low=as_float(require(payload, "l"), "l"),
        close=as_float(require(payload, "c"), "c"),
        volume=as_float(require(payload, "v"), "v"),
        interval_seconds=interval_seconds,
        vwap=as_float(vwap_raw, "vw") if vwap_raw is not None else None,
        trade_count=int(require(payload, "n", label="n (trade_count)")),
        adjusted=False,
    )