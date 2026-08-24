"""Alpaca adapter helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping


def iso_to_epoch(value: str) -> float:
    text = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(
            f"timestamp ISO inválido de Alpaca: {value!r}"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def require(
    payload: Mapping[str, Any],
    key: str,
    *,
    label: str | None = None,
) -> Any:
    if key not in payload:
        raise ValueError(
            f"payload de Alpaca sin campo '{key}' "
            f"({label or 'necesario para el tipo de evento'})"
        )
    return payload[key]


def as_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"campo '{field}' de Alpaca no es numérico: {value!r}"
        ) from exc