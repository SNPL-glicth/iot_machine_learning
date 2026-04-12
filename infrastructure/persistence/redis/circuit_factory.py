"""Circuit breaker factory functions.

Extracted from circuit.py as part of modularization.
Provides pre-configured circuit breakers for Redis operations.
"""

from __future__ import annotations

import logging
from typing import Dict

from .circuit_breaker import CircuitBreaker

logger = logging.getLogger(__name__)

# Registry of circuit breakers
_redis_circuit_breakers: Dict[str, CircuitBreaker] = {}


def get_redis_circuit_breaker(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for Redis operations.
    
    Pre-configured settings:
    - redis_broker: threshold=3, recovery=30s (fast recovery, critical path)
    - redis_cache: threshold=5, recovery=60s (tolerate more failures)
    - redis_window: threshold=3, recovery=30s (critical for ML)
    
    Args:
        name: Breaker name (redis_broker, redis_cache, redis_window)
        
    Returns:
        CircuitBreaker instance
    """
    if name not in _redis_circuit_breakers:
        configs = {
            "redis_broker": {
                "failure_threshold": 3,
                "recovery_timeout": 30,
            },
            "redis_cache": {
                "failure_threshold": 5,
                "recovery_timeout": 60,
            },
            "redis_window": {
                "failure_threshold": 3,
                "recovery_timeout": 30,
            },
        }
        
        config = configs.get(name, {
            "failure_threshold": 5,
            "recovery_timeout": 30,
        })
        
        _redis_circuit_breakers[name] = CircuitBreaker(
            name=name,
            **config
        )
    
    return _redis_circuit_breakers[name]


def reset_all_circuits() -> None:
    """Reset all circuit breakers (for testing)."""
    global _redis_circuit_breakers
    _redis_circuit_breakers.clear()
    logger.debug("all_circuit_breakers_reset")
