"""Standalone key-parsing utilities for Redis namespaced keys.

Keeps RedisNamespace focused on key *generation* while these functions handle
*reverse* parsing (extraction) and validation.
"""

from __future__ import annotations

from typing import Optional


def extract_env(namespaced_key: str) -> Optional[str]:
    """Extract env from namespaced key."""
    parts = namespaced_key.split(":")
    return parts[0] if parts else None


def extract_app(namespaced_key: str) -> Optional[str]:
    """Extract app from namespaced key."""
    parts = namespaced_key.split(":")
    return parts[1] if len(parts) > 1 else None


def extract_tenant_id(namespaced_key: str) -> Optional[str]:
    """Extract tenant_id from namespaced key."""
    parts = namespaced_key.split(":")
    if len(parts) >= 3:
        return parts[2]
    return None


def extract_resource_id(namespaced_key: str) -> Optional[str]:
    """Extract resource_id from namespaced key.

    Expected format::

        env:app:tenant[:context]:type:id[:suffix]

    With context: type=4, id=5
    Without context: type=3, id=4
    """
    parts = namespaced_key.split(":")
    if len(parts) >= 6 and parts[3] not in ("series", "regime", "weights"):
        return parts[5]
    if len(parts) >= 5:
        return parts[4]
    return None


def is_valid_key(
    key: str,
    env: str,
    app: str,
    tenant_id: str,
    context_type: str = "default",
) -> bool:
    """Check if key follows namespace format for the given parameters."""
    parts = key.split(":")
    # Minimum without context: env:app:tenant:type:id (5 parts)
    # Minimum with context: env:app:tenant:context:type:id (6 parts)
    if len(parts) < 5:
        return False

    if parts[0] != env or parts[1] != app or parts[2] != tenant_id:
        return False

    if context_type != "default":
        if len(parts) < 6:
            return False
        if parts[3] != context_type:
            return False

    return True
