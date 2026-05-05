"""Redis utilities and helpers.

Extracted from redis_connection_manager.py as part of modularization.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import time
from typing import Any, Callable, Optional, TypeVar

from .config import DEFAULT_REDIS_URL

logger = logging.getLogger(__name__)
T = TypeVar("T")


def redis_retry(
    max_attempts: int = 3,
    base_delay: float = 0.05,
    max_delay: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries Redis operations with exponential backoff + jitter.

    Args:
        max_attempts: Maximum number of attempts (including the first).
        base_delay: Initial delay between retries in seconds.
        max_delay: Cap on delay between retries.
        exceptions: Tuple of exception types to catch and retry.
        on_retry: Optional callback(exc, attempt_number) invoked on each retry.
    """

    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt >= max_attempts:
                        break
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    delay = delay * (0.8 + 0.4 * (hash(str(time.time())) % 1000) / 1000.0)
                    if on_retry:
                        on_retry(exc, attempt)
                    else:
                        logger.warning(
                            "redis_retry_attempt",
                            extra={
                                "function": fn.__name__,
                                "attempt": attempt,
                                "delay_ms": round(delay * 1000, 1),
                                "error": str(exc),
                            },
                        )
                    time.sleep(delay)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


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
