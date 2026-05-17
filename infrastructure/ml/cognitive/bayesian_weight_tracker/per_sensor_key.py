"""Per-sensor regime key construction — pure functions, no state, no I/O."""

from __future__ import annotations

from typing import Optional

COLD_START_THRESHOLD: int = 50


def build_regime_key(domain_namespace: str, regime: str, series_id: Optional[str] = None) -> str:
    """Build regime key with or without series_id."""
    if not series_id or series_id == "unknown":
        return f"{domain_namespace}:{regime}"
    return f"{domain_namespace}:{series_id}:{regime}"


def build_fallback_key(domain_namespace: str, regime: str) -> str:
    """Global key (no series_id) for fallback when no per-sensor history exists."""
    return f"{domain_namespace}:{regime}"


def should_use_per_sensor(
    series_id: Optional[str],
    accuracy_data: dict,
    domain_namespace: str,
    regime: str,
    threshold: int = COLD_START_THRESHOLD,
) -> bool:
    """True if enough per-sensor history exists to trust per-sensor weights."""
    if not series_id or series_id == "unknown":
        return False
    key = build_regime_key(domain_namespace, regime, series_id)
    if key not in accuracy_data:
        return False
    return len(accuracy_data[key]) >= threshold
