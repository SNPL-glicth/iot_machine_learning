"""Binance adapter helpers."""

from __future__ import annotations

from typing import Any, Mapping


def require(payload: Mapping[str, Any], key: str, *, label: str | None = None) -> Any:
    if key not in payload:
        raise ValueError(
            f"payload de Binance sin campo '{key}' "
            f"({label or 'necesario para el tipo de evento'})"
        )
    return payload[key]


def as_float(value: Any, field: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"campo '{field}' de Binance no es numérico: {value!r}"
        ) from exc


def ms_to_epoch(ms: Any, field: str = "T") -> float:
    return as_float(ms, field) / 1000.0