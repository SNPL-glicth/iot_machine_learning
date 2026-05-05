"""Redis Keys Base — core sanitization and key building logic.

Provides base functionality for namespace isolation and key construction.
"""

from __future__ import annotations
import re


class RedisKeysBase:
    """Base class for Redis key management with sanitization and namespace isolation.
    
    Attributes:
        env: Environment (production/staging/dev)
        tenant_id: Sanitized tenant identifier
        app_name: Application name (default: zenin_ml)
        default_ttl: Default TTL in seconds for keys without explicit TTL
    """
    
    # Key sanitization pattern: allow alphanumeric, underscore, hyphen
    _SAFE_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+$')
    _MAX_KEY_LENGTH = 200
    
    def __init__(
        self,
        env: str = "production",
        tenant_id: str = "default",
        app_name: str = "zenin_ml",
        default_ttl: int = 86400,  # 24 hours
    ):
        """Initialize RedisKeys with namespace isolation.
        
        Args:
            env: Environment identifier
            tenant_id: Tenant identifier (will be sanitized)
            app_name: Application name
            default_ttl: Default TTL for keys in seconds
        
        Raises:
            ValueError: If identifiers contain unsafe characters
        """
        self.env = self._sanitize(env, "env")
        self.tenant_id = self._sanitize(tenant_id, "tenant_id")
        self.app_name = self._sanitize(app_name, "app_name")
        self.default_ttl = default_ttl
    
    @classmethod
    def _sanitize(cls, value: str, field_name: str) -> str:
        """Sanitize identifier to prevent injection.
        
        Args:
            value: Raw identifier
            field_name: Field name for error messages
        
        Returns:
            Sanitized identifier
        
        Raises:
            ValueError: If identifier is invalid
        """
        if not value:
            raise ValueError(f"{field_name} cannot be empty")
        
        # Replace spaces and special chars with underscore
        sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', value)
        
        # Truncate if too long
        if len(sanitized) > 64:
            sanitized = sanitized[:64]
        
        # Ensure starts with alphanumeric
        if not sanitized[0].isalnum():
            sanitized = 'x' + sanitized[1:]
        
        return sanitized
    
    def _build_key(self, resource: str, *parts: str) -> str:
        """Build namespaced key with validation.
        
        Args:
            resource: Resource type (e.g., 'plasticity', 'error_history')
            *parts: Additional key components
        
        Returns:
            Fully namespaced key
        
        Raises:
            ValueError: If key exceeds max length
        """
        # Sanitize all parts
        sanitized_parts = [self._sanitize(str(p), f"part_{i}") for i, p in enumerate(parts)]
        
        # Build key: env:app:tenant:resource:parts
        components = [self.env, self.app_name, self.tenant_id, resource] + sanitized_parts
        key = ':'.join(components)
        
        if len(key) > self._MAX_KEY_LENGTH:
            raise ValueError(f"Key exceeds max length {self._MAX_KEY_LENGTH}: {key[:50]}...")
        
        return key
