"""Rate Limiter — production-grade DoS prevention.

Uses Redis-backed sliding window algorithm for accurate rate limiting.
Prevents logical DoS and resource abuse.

Algorithm: Sliding Window Counter
- Tracks requests in time windows
- Accurate rate limiting (no token bucket drift)
- Efficient Redis operations (ZSET + EXPIRE)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from iot_machine_learning.infrastructure.security.redis_namespace import (
    RedisNamespace,
    get_namespace,
)

logger = logging.getLogger(__name__)


class RateLimitScope(str, Enum):
    """Rate limit scope types."""
    TENANT = "tenant"
    SERIES = "series"
    ENDPOINT = "endpoint"
    GLOBAL = "global"


@dataclass(frozen=True)
class RateLimitConfig:
    """Rate limit configuration.
    
    Attributes:
        requests_per_second: Max requests per second
        burst_size: Max burst (requests allowed in short spike)
        window_seconds: Time window for sliding window (default 1s)
    """
    requests_per_second: int
    burst_size: int
    window_seconds: float = 1.0
    
    def __post_init__(self):
        if self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be > 0")
        if self.burst_size < self.requests_per_second:
            raise ValueError("burst_size must be >= requests_per_second")
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be > 0")


@dataclass(frozen=True)
class RateLimitResult:
    """Result of rate limit check.
    
    Attributes:
        allowed: Whether request is allowed
        remaining: Requests remaining in window
        reset_at: Unix timestamp when limit resets
        retry_after: Seconds to wait before retry (if blocked)
    """
    allowed: bool
    remaining: int
    reset_at: float
    retry_after: float = 0.0


class RateLimiter:
    """Redis-backed rate limiter using sliding window algorithm.
    
    Features:
    - Accurate sliding window (no drift)
    - Per-tenant, per-series, per-endpoint limits
    - Burst support
    - Automatic cleanup of old entries
    - Fail-open on Redis errors
    """
    
    def __init__(
        self,
        redis_client: Any,
        tenant_id: str = "default",
        namespace: Optional[RedisNamespace] = None,
        default_config: Optional[RateLimitConfig] = None,
    ) -> None:
        """Initialize rate limiter.
        
        Args:
            redis_client: Redis client
            tenant_id: Tenant ID for namespacing
            namespace: Optional pre-configured namespace
            default_config: Default rate limit config
        """
        self._redis = redis_client
        self._namespace = namespace or get_namespace(tenant_id=tenant_id)
        self._tenant_id = tenant_id
        
        # Default: 100 req/s, burst 150
        self._default_config = default_config or RateLimitConfig(
            requests_per_second=100,
            burst_size=150,
        )
        
        # Metrics
        self._blocked_count = 0
        self._allowed_count = 0
    
    def check_rate_limit(
        self,
        scope: RateLimitScope,
        identifier: str,
        config: Optional[RateLimitConfig] = None,
    ) -> RateLimitResult:
        """Check if request is allowed under rate limit.
        
        Args:
            scope: Rate limit scope (tenant, series, endpoint)
            identifier: Unique identifier for this scope
            config: Optional config override
        
        Returns:
            RateLimitResult with decision and metadata
        """
        cfg = config or self._default_config
        
        # Fail-open on Redis errors
        if self._redis is None:
            logger.warning("rate_limiter_redis_unavailable: allowing request")
            return RateLimitResult(
                allowed=True,
                remaining=cfg.burst_size,
                reset_at=time.time() + cfg.window_seconds,
            )
        
        try:
            now = time.time()
            window_start = now - cfg.window_seconds
            
            # Generate namespaced key
            key = self._namespace.key(
                resource_type=f"ratelimit_{scope.value}",
                resource_id=identifier,
            )
            
            # Use Redis pipeline for atomic operations
            pipe = self._redis.pipeline()
            
            # 1. Remove old entries outside window
            pipe.zremrangebyscore(key, 0, window_start)
            
            # 2. Count requests in current window
            pipe.zcard(key)
            
            # 3. Add current request with timestamp as score
            request_id = f"{now}:{id(self)}"
            pipe.zadd(key, {request_id: now})
            
            # 4. Set TTL (cleanup)
            pipe.expire(key, int(cfg.window_seconds * 2))
            
            # Execute pipeline
            results = pipe.execute()
            
            # Extract count (before adding current request)
            count_before = results[1]
            
            # Check against burst limit
            if count_before >= cfg.burst_size:
                # Remove the request we just added (over limit)
                self._redis.zrem(key, request_id)
                
                # Calculate retry_after
                oldest_in_window = self._redis.zrange(key, 0, 0, withscores=True)
                if oldest_in_window:
                    oldest_timestamp = oldest_in_window[0][1]
                    retry_after = (oldest_timestamp + cfg.window_seconds) - now
                else:
                    retry_after = cfg.window_seconds
                
                self._blocked_count += 1
                
                logger.warning(
                    "rate_limit_exceeded",
                    extra={
                        "scope": scope.value,
                        "identifier": identifier,
                        "count": count_before,
                        "limit": cfg.burst_size,
                    }
                )
                
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_at=now + cfg.window_seconds,
                    retry_after=max(0, retry_after),
                )
            
            # Request allowed
            self._allowed_count += 1
            remaining = cfg.burst_size - count_before - 1
            
            return RateLimitResult(
                allowed=True,
                remaining=max(0, remaining),
                reset_at=now + cfg.window_seconds,
            )
        
        except Exception as e:
            # Fail-open on errors
            logger.error(
                "rate_limiter_error: failing open",
                extra={"error": str(e), "scope": scope.value}
            )
            return RateLimitResult(
                allowed=True,
                remaining=cfg.burst_size,
                reset_at=time.time() + cfg.window_seconds,
            )
    
    def check_tenant_limit(
        self,
        config: Optional[RateLimitConfig] = None,
    ) -> RateLimitResult:
        """Check rate limit for current tenant.
        
        Args:
            config: Optional config override
        
        Returns:
            RateLimitResult
        """
        return self.check_rate_limit(
            scope=RateLimitScope.TENANT,
            identifier=self._tenant_id,
            config=config,
        )
    
    def check_series_limit(
        self,
        series_id: str,
        config: Optional[RateLimitConfig] = None,
    ) -> RateLimitResult:
        """Check rate limit for specific series.
        
        Args:
            series_id: Series identifier
            config: Optional config override
        
        Returns:
            RateLimitResult
        """
        return self.check_rate_limit(
            scope=RateLimitScope.SERIES,
            identifier=series_id,
            config=config,
        )
    
    def check_endpoint_limit(
        self,
        endpoint: str,
        config: Optional[RateLimitConfig] = None,
    ) -> RateLimitResult:
        """Check rate limit for specific endpoint.
        
        Args:
            endpoint: Endpoint name (e.g., 'predict', 'ingest')
            config: Optional config override
        
        Returns:
            RateLimitResult
        """
        return self.check_rate_limit(
            scope=RateLimitScope.ENDPOINT,
            identifier=endpoint,
            config=config,
        )
    
    def get_metrics(self) -> dict:
        """Get rate limiter metrics.
        
        Returns:
            Dict with blocked_count, allowed_count, block_rate
        """
        total = self._blocked_count + self._allowed_count
        block_rate = self._blocked_count / total if total > 0 else 0.0
        
        return {
            "blocked_count": self._blocked_count,
            "allowed_count": self._allowed_count,
            "total_requests": total,
            "block_rate": block_rate,
        }
    
    def reset_metrics(self) -> None:
        """Reset metrics counters."""
        self._blocked_count = 0
        self._allowed_count = 0


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded.
    
    Attributes:
        result: RateLimitResult with details
        scope: Rate limit scope
        identifier: Identifier that exceeded limit
    """
    
    def __init__(
        self,
        result: RateLimitResult,
        scope: RateLimitScope,
        identifier: str,
    ):
        self.result = result
        self.scope = scope
        self.identifier = identifier
        
        message = (
            f"Rate limit exceeded for {scope.value}:{identifier}. "
            f"Retry after {result.retry_after:.2f}s"
        )
        super().__init__(message)
