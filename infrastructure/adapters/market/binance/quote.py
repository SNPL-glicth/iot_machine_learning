"""Binance quote parser."""

from __future__ import annotations

from typing import Any, Mapping

from iot_machine_learning.domain.entities.market import Quote, DataStatus
from .helpers import require, as_float


def quote_from_payload(
    payload: Mapping[str, object],
    *,
    received_at: float | None = None,
    provider_name: str = "binance",
    profile_has_realtime: bool = True,
) -> Quote:
    if "s" not in payload or "b" not in payload:
        raise ValueError(
            "esperaba bookTicker de Binance (campos s/b/B/a/A), "
            f"obtenido {sorted(payload)}"
        )
    if received_at is None:
        raise ValueError(
            "bookTicker de Binance no lleva timestamp: "
            "received_at es requerido"
        )
    symbol = str(require(payload, "s"))
    bid = as_float(require(payload, "b"), "b")
    bid_size = as_float(require(payload, "B"), "B")
    ask = as_float(require(payload, "a"), "a")
    ask_size = as_float(require(payload, "A"), "A")
    data_status = DataStatus.REALTIME if profile_has_realtime else DataStatus.DELAYED
    return Quote(
        symbol=symbol,
        timestamp=received_at,
        data_status=data_status,
        source_provider=provider_name,
        venue=None,
        bid=bid,
        bid_size=bid_size,
        ask=ask,
        ask_size=ask_size,
        bid_exchange=None,
        ask_exchange=None,
        conditions=(),
        tape=None,
    )