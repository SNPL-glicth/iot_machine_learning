"""Tests for RateLimiter - DoS prevention."""

import time
import pytest
from unittest.mock import Mock, MagicMock

from iot_machine_learning.infrastructure.security.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitScope,
    RateLimitResult,
    RateLimitExceeded,
)


class TestRateLimitConfig:
    """Test RateLimitConfig validation."""
    
    def test_valid_config(self):
        """Valid config should work."""
        config = RateLimitConfig(
            requests_per_second=10,
            burst_size=20,
        )
        
        assert config.requests_per_second == 10
        assert config.burst_size == 20
        assert config.window_seconds == 1.0
    
    def test_invalid_requests_per_second(self):
        """requests_per_second must be > 0."""
        with pytest.raises(ValueError, match="requests_per_second must be > 0"):
            RateLimitConfig(requests_per_second=0, burst_size=10)
    
    def test_invalid_burst_size(self):
        """burst_size must be >= requests_per_second."""
        with pytest.raises(ValueError, match="burst_size must be >="):
            RateLimitConfig(requests_per_second=10, burst_size=5)
    
    def test_invalid_window_seconds(self):
        """window_seconds must be > 0."""
        with pytest.raises(ValueError, match="window_seconds must be > 0"):
            RateLimitConfig(
                requests_per_second=10,
                burst_size=20,
                window_seconds=0,
            )


class TestRateLimiter:
    """Test RateLimiter core functionality."""
    
    def setup_method(self):
        """Setup mock Redis client."""
        self.redis = Mock()
        self.redis.pipeline.return_value = self._create_mock_pipeline()
        
        self.limiter = RateLimiter(
            redis_client=self.redis,
            tenant_id="test_tenant",
            default_config=RateLimitConfig(
                requests_per_second=10,
                burst_size=15,
            ),
        )
    
    def _create_mock_pipeline(self):
        """Create mock Redis pipeline."""
        pipe = Mock()
        pipe.zremrangebyscore.return_value = None
        pipe.zcard.return_value = None
        pipe.zadd.return_value = None
        pipe.expire.return_value = None
        pipe.execute.return_value = [None, 0, None, None]  # count_before = 0
        return pipe
    
    def test_initialization(self):
        """Initialization should work."""
        assert self.limiter._tenant_id == "test_tenant"
        assert self.limiter._default_config.requests_per_second == 10
    
    def test_check_rate_limit_allowed(self):
        """Request should be allowed when under limit."""
        result = self.limiter.check_rate_limit(
            scope=RateLimitScope.SERIES,
            identifier="sensor_42",
        )
        
        assert result.allowed is True
        assert result.remaining >= 0
        assert result.retry_after == 0.0
    
    def test_check_rate_limit_exceeded(self):
        """Request should be blocked when over limit."""
        # Mock pipeline to return count >= burst_size
        pipe = self._create_mock_pipeline()
        pipe.execute.return_value = [None, 15, None, None]  # count_before = 15
        self.redis.pipeline.return_value = pipe
        self.redis.zrange.return_value = [(b"req1", time.time())]
        
        result = self.limiter.check_rate_limit(
            scope=RateLimitScope.SERIES,
            identifier="sensor_42",
        )
        
        assert result.allowed is False
        assert result.remaining == 0
        assert result.retry_after >= 0
    
    def test_check_tenant_limit(self):
        """Tenant limit check should work."""
        result = self.limiter.check_tenant_limit()
        
        assert result.allowed is True
    
    def test_check_series_limit(self):
        """Series limit check should work."""
        result = self.limiter.check_series_limit(series_id="sensor_42")
        
        assert result.allowed is True
    
    def test_check_endpoint_limit(self):
        """Endpoint limit check should work."""
        result = self.limiter.check_endpoint_limit(endpoint="predict")
        
        assert result.allowed is True
    
    def test_metrics(self):
        """Metrics should be tracked."""
        # Allow one request
        self.limiter.check_rate_limit(
            scope=RateLimitScope.SERIES,
            identifier="sensor_1",
        )
        
        # Block one request
        pipe = self._create_mock_pipeline()
        pipe.execute.return_value = [None, 15, None, None]
        self.redis.pipeline.return_value = pipe
        self.redis.zrange.return_value = [(b"req1", time.time())]
        
        self.limiter.check_rate_limit(
            scope=RateLimitScope.SERIES,
            identifier="sensor_2",
        )
        
        metrics = self.limiter.get_metrics()
        
        assert metrics["allowed_count"] == 1
        assert metrics["blocked_count"] == 1
        assert metrics["total_requests"] == 2
        assert metrics["block_rate"] == 0.5
    
    def test_reset_metrics(self):
        """Metrics should be resettable."""
        self.limiter.check_rate_limit(
            scope=RateLimitScope.SERIES,
            identifier="sensor_1",
        )
        
        self.limiter.reset_metrics()
        
        metrics = self.limiter.get_metrics()
        assert metrics["allowed_count"] == 0
        assert metrics["blocked_count"] == 0
    
    def test_fail_open_on_redis_error(self):
        """Should fail open on Redis errors."""
        self.redis.pipeline.side_effect = Exception("Redis down")
        
        result = self.limiter.check_rate_limit(
            scope=RateLimitScope.SERIES,
            identifier="sensor_42",
        )
        
        # Should allow request (fail-open)
        assert result.allowed is True
    
    def test_fail_open_on_no_redis(self):
        """Should fail open when Redis is None."""
        limiter = RateLimiter(
            redis_client=None,
            tenant_id="test",
        )
        
        result = limiter.check_rate_limit(
            scope=RateLimitScope.SERIES,
            identifier="sensor_42",
        )
        
        assert result.allowed is True


class TestRateLimitExceeded:
    """Test RateLimitExceeded exception."""
    
    def test_exception_creation(self):
        """Exception should be created with details."""
        result = RateLimitResult(
            allowed=False,
            remaining=0,
            reset_at=time.time() + 1,
            retry_after=1.5,
        )
        
        exc = RateLimitExceeded(
            result=result,
            scope=RateLimitScope.SERIES,
            identifier="sensor_42",
        )
        
        assert exc.result == result
        assert exc.scope == RateLimitScope.SERIES
        assert exc.identifier == "sensor_42"
        assert "Retry after 1.50s" in str(exc)
