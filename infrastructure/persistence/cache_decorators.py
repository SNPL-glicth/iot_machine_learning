"""Cache decorators for easy caching of function results.

DEPRECATED: This module is kept for backward compatibility.

MIGRATION NOTICE (2026-04-09):
- The cache_system package is being consolidated into the unified Redis layer
- Use RedisConnectionManager directly for new code
- These decorators will be removed in a future release
"""

from __future__ import annotations

import warnings

# Emit deprecation warning
warnings.warn(
    "cache_decorators is deprecated. Use RedisConnectionManager directly.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from cache_system for backward compatibility during migration
# These will be replaced with unified implementation in future release
from .cache_system import (
    set_global_cache,
    get_global_cache,
    cached,
    cached_async,
    cache_invalidate,
    memoize,
)

__all__ = [
    "set_global_cache",
    "get_global_cache",
    "cached",
    "cached_async",
    "cache_invalidate",
    "memoize",
]
