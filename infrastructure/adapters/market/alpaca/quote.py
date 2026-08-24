"""Alpaca quote parser."""

from __future__ import annotations

from typing import Any, Mapping

from iot_machine_learning.domain.entities.market import Quote, DataStatus
from .helpers import require, as_float, iso_to_epoch


def quote_from_payload(
    payload: Mapping[str, object],
    *,
    received_at: float | None = None,
    provider_name: str = "alpaca",
    profile_has_realtime: bool = True,
) -> Quote:
    if payload.get("T") != "q":
        raise ValueError(
            f"esperaba evento 'q' de Alpaca, obtenido {payload.get('T')!r}"
        )
    symbol = str(require(payload, "S"))
    timestamp = iso_to_epoch(str(require(payload, "t")))
    conditions = tuple(
        str(cond) for cond in require(payload, "c", label="c (conditions)")
    )
    tape = str(require(payload, "z", label="z (tape)"))
    data_status = DataStatus.REALTIME if profile_has_realtime else DataStatus.DELAYED
    return Quote(
        symbol=symbol,
        timestamp=timestamp,
        data_status=data_status,
        source_provider=provider_name,
        venue=str(require(payload, "bx", label="bx (venue)")),
        bid=as_float(require(payload, "bp"), "bp"),
        bid_size=as_float(require(payload, "bs"), "bs"),
        ask=as_float(require(payload, "ap"), "ap"),
        ask_size=as_float(require(payload, "as"), "as"),
        bid_exchange=str(require(payload, "bx", label="bx")),
        ask_exchange=str(require(payload, "ax", label="ax")),
        conditions=conditions,
        tape=tape,
    )