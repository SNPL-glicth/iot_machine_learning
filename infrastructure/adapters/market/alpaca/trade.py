"""Alpaca trade parser."""

from __future__ import annotations

from typing import Any, Mapping

from iot_machine_learning.domain.entities.market import Trade, DataStatus
from .helpers import require, as_float, iso_to_epoch


def trade_from_payload(
    payload: Mapping[str, object],
    *,
    received_at: float | None = None,
    provider_name: str = "alpaca",
    profile_has_realtime: bool = True,
) -> Trade:
    if payload.get("T") != "t":
        raise ValueError(
            f"esperaba evento 't' de Alpaca, obtenido {payload.get('T')!r}"
        )
    symbol = str(require(payload, "S"))
    price = as_float(require(payload, "p"), "p")
    size = as_float(require(payload, "s"), "s")
    timestamp = iso_to_epoch(str(require(payload, "t")))
    trade_id = str(require(payload, "i"))
    conditions = tuple(
        str(cond) for cond in require(payload, "c", label="c (conditions)")
    )
    tape = str(require(payload, "z", label="z (tape)"))
    corrected = bool(require(payload, "u", label="u (corrected)"))
    data_status = DataStatus.REALTIME if profile_has_realtime else DataStatus.DELAYED
    return Trade(
        symbol=symbol,
        timestamp=timestamp,
        data_status=data_status,
        source_provider=provider_name,
        venue=str(require(payload, "x", label="x (venue)")),
        price=price,
        size=size,
        trade_id=trade_id,
        taker_side=None,
        conditions=conditions,
        tape=tape,
        corrected=corrected,
    )