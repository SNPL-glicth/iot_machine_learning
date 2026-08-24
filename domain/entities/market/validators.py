"""Validadores compartidos del dominio de mercado (FASE 2/3).

Funciones puras de saneamiento/validación reutilizadas por las
observaciones (market) y la predicción (prediction). Mensajes estables:
los tests dependen de ellos (ej. "horizon", "símbolo").
"""

from __future__ import annotations

import math


def validate_symbol(symbol: str) -> str:
    """Normaliza y valida un símbolo (no vacío, sin espacios)."""
    symbol = symbol.strip()
    if not symbol:
        raise ValueError("symbol no puede ser vacío")
    if any(ch.isspace() for ch in symbol):
        raise ValueError(f"symbol no puede contener espacios: {symbol!r}")
    return symbol


def validate_non_empty_str(value: str, name: str) -> str:
    """Normaliza y valida una cadena opcional no vacía."""
    value = value.strip()
    if not value:
        raise ValueError(f"{name} no puede ser vacío")
    return value


def validate_timestamp(timestamp: float) -> None:
    """Valida un timestamp epoch (finito, > 0)."""
    if not math.isfinite(timestamp) or timestamp <= 0:
        raise ValueError(f"timestamp inválido: {timestamp!r}")


def validate_price(price: float, name: str) -> None:
    """Valida un precio (finito, > 0)."""
    if not math.isfinite(price) or price <= 0:
        raise ValueError(f"{name} debe ser finito y > 0: {price!r}")


def validate_size(size: float, name: str) -> None:
    """Valida una cantidad (finito, >= 0)."""
    if not math.isfinite(size) or size < 0:
        raise ValueError(f"{name} debe ser finito y >= 0: {size!r}")


def validate_unit_interval(value: float, name: str) -> None:
    """Valida una probabilidad/confianza (finito, [0, 1])."""
    if not math.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} fuera de [0, 1]: {value!r}")
