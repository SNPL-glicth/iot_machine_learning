"""Factory functions for creating RedisNamespace instances.

Provides cached and uncached constructors to centralise namespace creation
and avoid scattered ``RedisNamespace(...)`` calls across the codebase.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from .redis_namespace import RedisNamespace


@lru_cache(maxsize=128)
def get_namespace(
    tenant_id: str,
    env: Optional[str] = None,
    context_type: str = "default",
) -> RedisNamespace:
    """Get or create cached RedisNamespace.

    Args:
        tenant_id: Tenant ID (required)
        env: Environment (optional, auto-detected)
        context_type: Data context type ("numeric", "documental", "default")

    Returns:
        Cached RedisNamespace instance
    """
    return RedisNamespace(tenant_id=tenant_id, env=env, context_type=context_type)


def create_namespace(
    tenant_id: str,
    env: Optional[str] = None,
    app: str = "zenin",
    context_type: str = "default",
    default_ttl: int = 86400,
    strict_mode: bool = True,
) -> RedisNamespace:
    """Factory for RedisNamespace.

    Args:
        tenant_id: Tenant ID (REQUIRED)
        env: Environment
        app: Application name
        context_type: Data context type ("numeric", "documental", "default")
        default_ttl: Default TTL
        strict_mode: Strict validation

    Returns:
        RedisNamespace instance
    """
    return RedisNamespace(
        tenant_id=tenant_id,
        env=env,
        app=app,
        context_type=context_type,
        default_ttl=default_ttl,
        strict_mode=strict_mode,
    )
