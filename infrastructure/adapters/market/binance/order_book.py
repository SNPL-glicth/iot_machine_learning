"""Binance order book parser."""

from __future__ import annotations

from typing import Any, Mapping

from iot_machine_learning.domain.entities.market import OrderBookSnapshot, DataStatus
from .helpers import require, as_float


def order_book_from_payload(
    payload: Mapping[str, object],
    *,
    symbol: str | None = None,
    received_at: float | None = None,
    provider_name: str = "binance",
    profile_has_realtime: bool = True,
) -> OrderBookSnapshot:
    if "lastUpdateId" not in payload:
        raise ValueError(
            f"esperaba snapshot de depth de Binance "
            f"(campos lastUpdateId/bids/asks), obtenido {sorted(payload)}"
        )
    if received_at is None:
        raise ValueError(
            "depth de Binance no lleva timestamp: "
            "received_at es requerido"
        )
    payload_symbol = payload.get("s")
    if payload_symbol is None and symbol is None:
        raise ValueError(
            "symbol requerido: el snapshot de depth de Binance no "
            "declara el símbolo en el cuerpo del payload"
        )
    resolved_symbol = str(payload_symbol if payload_symbol is not None else symbol)
    bids_raw = require(payload, "bids", label="bids")
    asks_raw = require(payload, "asks", label="asks")
    if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
        raise TypeError("bids/asks de Binance deben ser listas")
    bids = tuple(
        (as_float(level[0], "bid price"), as_float(level[1], "bid size"))
        for level in bids_raw
    )
    asks = tuple(
        (as_float(level[0], "ask price"), as_float(level[1], "ask size"))
        for level in asks_raw
    )
    if not bids and not asks:
        raise ValueError("deep de Binance con libro vacío")
    data_status = DataStatus.REALTIME if profile_has_realtime else DataStatus.DELAYED
    return OrderBookSnapshot(
        symbol=resolved_symbol,
        timestamp=received_at,
        data_status=data_status,
        source_provider=provider_name,
        venue=None,
        bids=bids,
        asks=asks,
        reset=True,
    )