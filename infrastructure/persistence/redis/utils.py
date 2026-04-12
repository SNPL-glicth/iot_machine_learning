"""Redis utilities and helpers.

Extracted from redis_connection_manager.py as part of modularization.
"""

from __future__ import annotations

import asyncio
import os

from .config import DEFAULT_REDIS_URL


def _detect_async_context() -> bool:
    """Detect if code is running inside an async event loop.
    
    Returns:
        True if inside async context (event loop running)
        False if in pure sync context
    """
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _get_redis_url() -> str:
    """Get Redis URL from environment with fallback."""
    return os.getenv("REDIS_URL", DEFAULT_REDIS_URL)
