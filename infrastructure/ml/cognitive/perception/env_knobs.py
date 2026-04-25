"""Environment-sourced configuration knobs for perception helpers."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        logger.warning("invalid_env_int", extra={"name": name, "value": raw})
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        logger.warning("invalid_env_float", extra={"name": name, "value": raw})
        return default


ML_PREDICT_MAX_WORKERS: int = _env_int("ML_PREDICT_MAX_WORKERS", 3)
ML_PREDICT_ENGINE_TIMEOUT_MS: float = _env_float(
    "ML_PREDICT_ENGINE_TIMEOUT_MS", 400.0
)
ML_ENGINE_TIMEOUT_MS: float = _env_float("ML_ENGINE_TIMEOUT_MS", 200.0)
