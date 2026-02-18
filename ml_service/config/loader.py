"""Feature flags loader from environment variables and singleton management."""

from __future__ import annotations

import logging
import os
from typing import Optional

from .flags import FeatureFlags
from .parsers import parse_bool

logger = logging.getLogger(__name__)


def load_from_env() -> FeatureFlags:
    """Carga feature flags desde variables de entorno.

    Variables reconocidas (todas opcionales):
    - ``ML_ROLLBACK_TO_BASELINE``: "true"/"false"
    - ``ML_USE_TAYLOR_PREDICTOR``: "true"/"false"
    - ``ML_USE_KALMAN_FILTER``: "true"/"false"
    - ``ML_TAYLOR_ORDER``: "1"/"2"/"3"
    - ``ML_TAYLOR_HORIZON``: "1"+ 
    - ``ML_KALMAN_Q``: float como string
    - ``ML_KALMAN_WARMUP_SIZE``: int como string
    - ``ML_ENABLE_AB_TESTING``: "true"/"false"
    - ``ML_TAYLOR_SENSOR_WHITELIST``: "1,5,42"
    - ``ML_DEFAULT_ENGINE``: nombre del motor

    Returns:
        ``FeatureFlags`` con valores de env vars o defaults seguros.
    """
    kwargs: dict = {}

    # Booleans
    for key in (
        "ML_ROLLBACK_TO_BASELINE",
        "ML_USE_TAYLOR_PREDICTOR",
        "ML_USE_KALMAN_FILTER",
        "ML_ENABLE_AB_TESTING",
        "ML_ENABLE_COGNITIVE_MEMORY",
        "ML_COGNITIVE_MEMORY_DRY_RUN",
        "ML_COGNITIVE_MEMORY_ASYNC",
        "ML_ENABLE_MEMORY_RECALL",
    ):
        env_val = os.environ.get(key)
        if env_val is not None:
            kwargs[key] = parse_bool(env_val)

    # Integers
    for key in ("ML_TAYLOR_ORDER", "ML_TAYLOR_HORIZON", "ML_KALMAN_WARMUP_SIZE"):
        env_val = os.environ.get(key)
        if env_val is not None:
            try:
                kwargs[key] = int(env_val.strip())
            except ValueError:
                logger.warning(
                    "feature_flag_parse_error",
                    extra={"field": key, "invalid_value": env_val},
                )

    # Floats
    for key in ("ML_KALMAN_Q",):
        env_val = os.environ.get(key)
        if env_val is not None:
            try:
                kwargs[key] = float(env_val.strip())
            except ValueError:
                logger.warning(
                    "feature_flag_parse_error",
                    extra={"field": key, "invalid_value": env_val},
                )

    # Batch enterprise booleans
    for key in ("ML_BATCH_USE_ENTERPRISE",):
        env_val = os.environ.get(key)
        if env_val is not None:
            kwargs[key] = parse_bool(env_val)

    # Strings
    for key in (
        "ML_TAYLOR_SENSOR_WHITELIST",
        "ML_DEFAULT_ENGINE",
        "ML_COGNITIVE_MEMORY_URL",
        "ML_BATCH_ENTERPRISE_SENSORS",
        "ML_BATCH_BASELINE_ONLY_SENSORS",
    ):
        env_val = os.environ.get(key)
        if env_val is not None:
            kwargs[key] = env_val.strip()

    flags = FeatureFlags(**kwargs)

    logger.info(
        "feature_flags_loaded",
        extra={
            "source": "env",
            "rollback": flags.ML_ROLLBACK_TO_BASELINE,
            "taylor": flags.ML_USE_TAYLOR_PREDICTOR,
            "kalman": flags.ML_USE_KALMAN_FILTER,
            "ab_testing": flags.ML_ENABLE_AB_TESTING,
            "default_engine": flags.ML_DEFAULT_ENGINE,
            "cognitive_memory": flags.ML_ENABLE_COGNITIVE_MEMORY,
            "cognitive_memory_dry_run": flags.ML_COGNITIVE_MEMORY_DRY_RUN,
        },
    )

    return flags


# --- Singleton global (lazy) ---
_global_flags: Optional[FeatureFlags] = None


def get_feature_flags() -> FeatureFlags:
    """Retorna la instancia global de feature flags (lazy singleton).

    La primera llamada carga desde env vars.  Llamadas subsiguientes
    retornan la misma instancia.

    Returns:
        ``FeatureFlags`` global.
    """
    global _global_flags
    if _global_flags is None:
        _global_flags = load_from_env()
    return _global_flags


def reset_feature_flags() -> None:
    """Resetea el singleton (para testing)."""
    global _global_flags
    _global_flags = None
