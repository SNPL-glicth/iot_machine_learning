"""Redis configuration and constants.

Extracted from redis_connection_manager.py as part of modularization.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    """Lee entero desde env; si es inválido, loguea warning y usa default."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning(
            "redis_config_env_invalid",
            extra={"env_var": name, "raw_value": raw, "using_default": default},
        )
        return default


# Connection defaults
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_SOCKET_TIMEOUT = 5.0
DEFAULT_SOCKET_CONNECT_TIMEOUT = 5.0

# FIX P1-1: Pool sizing configurables por env
GENERAL_MAX_CONNECTIONS = _env_int("REDIS_MAX_CONNECTIONS", 150)
STREAM_MAX_CONNECTIONS = _env_int("REDIS_STREAM_MAX_CONNECTIONS", 50)
DEFAULT_MAX_CONNECTIONS = 50  # Legacy compatibility

logger.info(
    "redis_pool_sizes_configured",
    extra={
        "general_max_connections": GENERAL_MAX_CONNECTIONS,
        "stream_max_connections": STREAM_MAX_CONNECTIONS,
        "env_redis_max_connections": os.getenv("REDIS_MAX_CONNECTIONS", "<unset>"),
        "env_redis_stream_max_connections": os.getenv("REDIS_STREAM_MAX_CONNECTIONS", "<unset>"),
    },
)
