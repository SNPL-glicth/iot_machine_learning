"""Broker factory for ML service.

Creates the appropriate broker based on configuration.
Supports fallback from Redis to InMemory if Redis is unavailable.
"""

from __future__ import annotations

import os
import logging
from enum import Enum
from typing import Optional

from ..reading_broker import ReadingBroker
from ..in_memory_broker import InMemoryReadingBroker
from .redis_reading_broker import RedisReadingBroker

logger = logging.getLogger(__name__)


class BrokerType(Enum):
    """Available broker types."""
    IN_MEMORY = "in_memory"
    REDIS = "redis"
    AUTO = "auto"  # Try Redis, fallback to InMemory


_broker_instance: Optional[ReadingBroker] = None


def create_broker(
    broker_type: Optional[BrokerType] = None,
    redis_url: Optional[str] = None,
    fallback_to_memory: bool = True,
) -> ReadingBroker:
    """Create a broker instance.
    
    Args:
        broker_type: Type of broker to create. Default: AUTO (from env or Redis with fallback)
        redis_url: Redis URL for Redis broker. Default: REDIS_URL env var
        fallback_to_memory: If True, fallback to InMemory if Redis fails
        
    Returns:
        ReadingBroker instance
        
    Environment Variables:
        BROKER_TYPE: "redis", "in_memory", or "auto"
        REDIS_URL: Redis connection URL
    """
    if broker_type is None:
        env_type = os.getenv("BROKER_TYPE", "auto").lower()
        broker_type = BrokerType(env_type) if env_type in ("redis", "in_memory", "auto") else BrokerType.AUTO
    
    if broker_type == BrokerType.IN_MEMORY:
        logger.info("[BROKER_FACTORY] Creating InMemoryReadingBroker")
        return InMemoryReadingBroker()
    
    if broker_type == BrokerType.REDIS:
        logger.info("[BROKER_FACTORY] Creating RedisReadingBroker")
        return RedisReadingBroker(redis_url=redis_url)
    
    # AUTO: Try Redis, fallback to InMemory
    logger.info("[BROKER_FACTORY] AUTO mode: trying Redis first")
    
    try:
        redis_broker = RedisReadingBroker(redis_url=redis_url)
        # Test connection
        health = redis_broker.health_check()
        if health.get("connected"):
            logger.info("[BROKER_FACTORY] Redis broker connected successfully")
            return redis_broker
        else:
            raise ConnectionError(health.get("last_error", "Unknown error"))
    except Exception as e:
        if fallback_to_memory:
            logger.warning(
                "[BROKER_FACTORY] Redis unavailable (%s), falling back to InMemory",
                str(e),
            )
            return InMemoryReadingBroker()
        else:
            raise


def get_broker() -> ReadingBroker:
    """Get the singleton broker instance.
    
    Creates the broker on first call and reuses it afterwards.
    """
    global _broker_instance
    if _broker_instance is None:
        _broker_instance = create_broker()
    return _broker_instance


def reset_broker() -> None:
    """Reset the singleton broker (for testing)."""
    global _broker_instance
    if _broker_instance is not None:
        if hasattr(_broker_instance, "stop"):
            _broker_instance.stop()
    _broker_instance = None


def get_broker_health() -> dict:
    """Get health status of the current broker."""
    broker = get_broker()
    if hasattr(broker, "health_check"):
        return broker.health_check()
    return {
        "type": type(broker).__name__,
        "connected": True,
        "note": "InMemory broker has no health check",
    }
