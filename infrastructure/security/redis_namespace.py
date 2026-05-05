"""Redis Namespacing — tenant isolation and key safety.

Prevents data leakage between tenants by enforcing namespace prefixes.
All Redis keys must go through this module.

Format: {env}:{app}:{tenant}:{context}:{resource_type}:{resource_id}[:suffix]
Example: prod:zenin:tenantA:numeric:series:sensor_123:sliding_window
"""

from __future__ import annotations

import hashlib
import logging
import os
from functools import lru_cache
from typing import Optional

from .namespace_validators import validate_and_sanitize

logger = logging.getLogger(__name__)


class RedisNamespace:
    """Enforces tenant isolation in Redis keys.
    
    Format: {env}:{app}:{tenant}:{context}:{resource_type}:{resource_id}[:suffix]
    
    Attributes:
        env: Environment (prod, staging, dev)
        app: Application name (zenin)
        tenant_id: Tenant identifier (REQUIRED for multi-tenant)
        context_type: Data context type ("numeric", "documental", "default")
        default_ttl: Default TTL for keys (seconds)
    """
    
    def __init__(
        self,
        tenant_id: str,
        env: Optional[str] = None,
        app: str = "zenin",
        context_type: str = "default",
        default_ttl: int = 86400,
        strict_mode: bool = True,
    ) -> None:
        """Initialize namespace. Raises ValueError on invalid tenant_id when strict_mode=True."""
        if not tenant_id:
            raise ValueError("tenant_id is REQUIRED. Use 'default' for single-tenant mode.")
        self._env = env or os.getenv("ZENIN_ENV", "prod")
        self._app = app
        self._tenant_id = self._validate_and_sanitize(tenant_id, "tenant_id", strict_mode)
        self._context_type = self._validate_and_sanitize(context_type, "context_type", strict_mode)
        self._default_ttl = default_ttl
        self._strict_mode = strict_mode
        
        # Cache para keys generadas (evita concatenaciones repetidas)
        self._key_cache: dict[tuple, str] = {}
    
    @staticmethod
    def _validate_and_sanitize(value: str, field_name: str, strict: bool) -> str:
        """Validate and optionally sanitize ID."""
        return validate_and_sanitize(value, field_name, strict)
    
    def key(
        self,
        resource_type: str,
        resource_id: str,
        suffix: Optional[str] = None,
    ) -> str:
        """Generate namespaced Redis key (cached)."""
        resource_type = self._validate_and_sanitize(
            resource_type, "resource_type", self._strict_mode
        )
        resource_id = self._validate_and_sanitize(
            resource_id, "resource_id", self._strict_mode
        )
        cache_key = (resource_type, resource_id, suffix, self._context_type)
        if cache_key in self._key_cache:
            return self._key_cache[cache_key]
        parts = [self._env, self._app, self._tenant_id]
        if self._context_type != "default":
            parts.append(self._context_type)
        parts.extend([resource_type, resource_id])
        if suffix:
            suffix = self._validate_and_sanitize(suffix, "suffix", self._strict_mode)
            parts.append(suffix)
        key = ":".join(parts)
        if len(self._key_cache) < 1000:
            self._key_cache[cache_key] = key
        return key
    
    def hash_key(self, resource_type: str, resource_id: str) -> str:
        """Hash key alias."""
        return self.key(resource_type, resource_id)
    
    def pattern(
        self,
        resource_type: str,
        pattern: str = "*",
    ) -> str:
        """Generate namespaced pattern for SCAN."""
        resource_type = self._validate_and_sanitize(
            resource_type, "resource_type", self._strict_mode
        )
        parts = [
            self._env,
            self._app,
            self._tenant_id,
        ]
        if self._context_type != "default":
            parts.append(self._context_type)
        parts.append(resource_type)
        return ":".join(parts) + f":{pattern}"

    def extract_resource_id(self, namespaced_key: str) -> Optional[str]:
        from .key_parsing import extract_resource_id as _e; return _e(namespaced_key)

    def extract_tenant_id(self, namespaced_key: str) -> Optional[str]:
        from .key_parsing import extract_tenant_id as _e; return _e(namespaced_key)

    def is_valid_key(self, key: str) -> bool:
        from .key_parsing import is_valid_key as _v
        return _v(key, env=self._env, app=self._app, tenant_id=self._tenant_id, context_type=self._context_type)
    
    @property
    def tenant_id(self) -> str: return self._tenant_id
    @property
    def context_type(self) -> str: return self._context_type
    @property
    def env(self) -> str: return self._env
    @property
    def app(self) -> str: return self._app
    @property
    def default_ttl(self) -> int: return self._default_ttl
    def clear_cache(self) -> None: self._key_cache.clear()

from .namespace_factory import create_namespace, get_namespace  # noqa: F401, E402
