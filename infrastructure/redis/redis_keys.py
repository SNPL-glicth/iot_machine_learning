"""Redis Key Registry — centralized key pattern management with strict isolation.

ISO 27001 A.12.4.1: All Redis key patterns are centralized and auditable.
DRY: Single source of truth for all Redis keys in the system.

Namespace format: {env}:{app}:{tenant}:{resource}
- env: production/staging/dev
- app: zenin_ml
- tenant: sanitized tenant identifier
- resource: specific key pattern

Usage:
    from iot_machine_learning.infrastructure.redis_keys import RedisKeys
    keys = RedisKeys(env="production", tenant_id="acme_corp")
    key = keys.plasticity("VOLATILE")
    key = keys.last_alert("series_123")
    
Modularized:
    - redis_keys_base.py: Core sanitization and key building
    - redis_keys_registry.py: Specific key patterns and TTLs
    - redis_keys.py: Facade (this file)
"""

from __future__ import annotations

# Import from modularized components
from .redis_keys_registry import RedisKeysRegistry


# Facade: expose RedisKeysRegistry as RedisKeys for backward compatibility
class RedisKeys(RedisKeysRegistry):
    """Registro central de todos los key patterns de Redis con aislamiento estricto.

    Modificar aquí afecta a todo el sistema. Auditables.
    
    This is a facade that inherits all functionality from RedisKeysRegistry.
    
    Attributes:
        env: Environment (production/staging/dev)
        tenant_id: Sanitized tenant identifier
        app_name: Application name (default: zenin_ml)
        default_ttl: Default TTL in seconds for keys without explicit TTL
    """
    pass


# Re-export for convenience
__all__ = ["RedisKeys"]
