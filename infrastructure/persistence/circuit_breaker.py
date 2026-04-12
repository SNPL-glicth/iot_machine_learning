"""Circuit breaker — Re-export from modular redis package.

DEPRECATED: This module is kept for backward compatibility.
Import from the modular package instead:
    from iot_machine_learning.infrastructure.persistence.redis import (
        CircuitBreaker,
        CircuitState,
        get_redis_circuit_breaker,
    )
"""

from __future__ import annotations

import warnings

warnings.warn(
    "circuit_breaker.py is deprecated. "
    "Import from infrastructure.persistence.redis instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export from modular package
from .redis.circuit import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    get_redis_circuit_breaker,
)

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "CircuitOpenError",
    "get_redis_circuit_breaker",
]
