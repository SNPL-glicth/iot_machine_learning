"""Parsing utilities for feature flags from environment variables."""

from __future__ import annotations

import logging
from typing import Optional, Set

logger = logging.getLogger(__name__)


def parse_bool(value: str) -> bool:
    """Parsea string a bool (case-insensitive).

    Args:
        value: String a parsear.

    Returns:
        ``True`` para "true", "1", "yes", "on".
        ``False`` para todo lo demás.
    """
    return value.strip().lower() in ("true", "1", "yes", "on")


def parse_int_set(value: Optional[str]) -> Set[int]:
    """Parsea string CSV de enteros a set.

    Args:
        value: String como ``"1,5,42"`` o ``None``.

    Returns:
        Set de enteros.  Vacío si ``value`` es ``None`` o vacío.
    """
    if not value or not value.strip():
        return set()

    result: Set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if part:
            try:
                result.add(int(part))
            except ValueError:
                logger.warning(
                    "feature_flag_parse_error",
                    extra={"field": "sensor_whitelist", "invalid_value": part},
                )
    return result
