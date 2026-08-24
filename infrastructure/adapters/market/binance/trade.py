"""Binance trade parser."""

from __future__ import annotations

from typing import Any, Mapping

from iot_machine_learning.domain.entities.market import Trade, DataStatus
from .helpers import require, as_float, ms_to_epoch


def trade_from_payload(
    payload: Mapping[str, object],
    *,
    received_at: float | None = None,
    provider_name: str = "binance",
    profile_has_realtime: bool = True,
) -> Trade:
    if payload.get("e") != "aggTrade":
        raise ValueError(
            f"esperaba evento 'aggTrade' de Binance, "
            f"obtenido {payload.get('e')!r}"
        )
    symbol = str(require(payload, "s"))
    price = as_float(require(payload, "p"), "p")
    size = as_float(require(payload, "q"), "q")
    timestamp = ms_to_epoch(require(payload, "T"), "T")
    trade_id = str(require(payload, "a", label="a (aggregate trade id)"))
    buyer_is_maker = bool(require(payload, "m", label="m (buyer is maker)"))
    data_status = DataStatus.REALTIME if profile_has_realtime else DataStatus.DELAYED
    return Trade(
        symbol=symbol,
        timestamp=timestamp,
        data_status=data_status,
        source_provider=provider_name,
        venue=None,
        price=price,
        size=size,
        trade_id=trade_id,
        taker_side="sell" if buyer_is_maker else "buy",
        conditions=(),
        tape=None,
        corrected=False,
    )