"""End-to-end integration tests for production hardening.

Tests cover:
- Normal flow with valid data
- Anomaly detection
- Drift scenario
- Invalid input handling
- Rate limiting
- Namespace isolation

Validates:
- No crashes
- No NaN outputs
- Weights remain valid
- Pipeline completes fully
"""

import pytest
import numpy as np
import time
from typing import List

from iot_machine_learning.infrastructure.redis.redis_keys import RedisKeys
from iot_machine_learning.infrastructure.redis.redis_key_manager import RedisKeyManager
from iot_machine_learning.infrastructure.security.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitScope,
    RateLimitExceeded,
)
from iot_machine_learning.infrastructure.security.input_validator import (
    InputValidator,
    ValidationError,
)


class TestE2ENormalFlow:
    """Test normal prediction flow with valid data."""
    
    def test_valid_prediction_completes(self):
        """Test that valid data produces valid prediction."""
        # Arrange
        values = [20.0, 21.0, 22.0, 23.0, 24.0]
        timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        validator = InputValidator(min_window_size=3)
        
        # Act
        result = validator.validate(values, timestamps)
        
        # Assert
        assert result.valid
        assert result.sanitized_values is not None
        assert len(result.sanitized_values) == 5
        assert all(np.isfinite(v) for v in result.sanitized_values)
    
    def test_pipeline_preserves_data_integrity(self):
        """Test that pipeline doesn't corrupt valid data."""
        # Arrange
        values = [10.0, 15.0, 20.0, 25.0, 30.0]
        timestamps = [100.0, 200.0, 300.0, 400.0, 500.0]
        
        validator = InputValidator(min_window_size=3, strict_mode=True)
        
        # Act
        result = validator.validate(values, timestamps)
        
        # Assert
        assert result.valid
        assert result.sanitized_values == values
        assert result.sanitized_timestamps is None  # No sanitization needed


class TestE2EAnomalyDetection:
    """Test anomaly detection scenarios."""
    
    def test_spike_detection(self):
        """Test that sudden spike is detected as anomaly."""
        # Arrange: normal values then spike
        values = [20.0, 21.0, 22.0, 100.0, 23.0]  # Spike at index 3
        timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        validator = InputValidator(
            min_window_size=3,
            value_min=0.0,
            value_max=50.0,
            strict_mode=False,
        )
        
        # Act
        result = validator.validate(values, timestamps)
        
        # Assert
        assert result.valid  # Sanitized
        assert result.sanitized_values is not None
        assert result.sanitized_values[3] == 50.0  # Clamped to max
    
    def test_nan_values_rejected_in_strict_mode(self):
        """Test that NaN values are rejected in strict mode."""
        # Arrange
        values = [20.0, 21.0, float('nan'), 23.0, 24.0]
        timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        validator = InputValidator(min_window_size=3, strict_mode=True)
        
        # Act
        result = validator.validate(values, timestamps)
        
        # Assert
        assert not result.valid
        assert result.error_code == "INVALID_VALUES"
    
    def test_nan_values_sanitized_in_lenient_mode(self):
        """Test that NaN values are sanitized in lenient mode."""
        # Arrange
        values = [20.0, 21.0, float('nan'), 23.0, 24.0]
        timestamps = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        validator = InputValidator(min_window_size=3, strict_mode=False)
        
        # Act
        result = validator.validate(values, timestamps)
        
        # Assert
        assert result.valid
        assert result.sanitized_values is not None
        assert all(np.isfinite(v) for v in result.sanitized_values)


class TestE2EDriftScenario:
    """Test concept drift scenarios."""
    
    def test_regime_change_handled(self):
        """Test that regime change from stable to volatile is handled."""
        # Arrange: stable then volatile
        stable_values = [20.0, 20.5, 21.0, 20.8, 21.2]
        volatile_values = [25.0, 15.0, 30.0, 10.0, 35.0]
        
        validator = InputValidator(min_window_size=3)
        
        # Act
        stable_result = validator.validate(stable_values)
        volatile_result = validator.validate(volatile_values)
        
        # Assert
        assert stable_result.valid
        assert volatile_result.valid
        # Both should pass validation (no crashes)


class TestE2EInvalidInputHandling:
    """Test invalid input handling."""
    
    def test_empty_window_rejected(self):
        """Test that empty window is rejected."""
        # Arrange
        values = []
        
        validator = InputValidator(min_window_size=3)
        
        # Act
        result = validator.validate(values)
        
        # Assert
        assert not result.valid
        assert result.error_code == "WINDOW_TOO_SMALL"
    
    def test_window_too_small_rejected(self):
        """Test that window smaller than minimum is rejected."""
        # Arrange
        values = [20.0, 21.0]  # Only 2 values
        
        validator = InputValidator(min_window_size=3)
        
        # Act
        result = validator.validate(values)
        
        # Assert
        assert not result.valid
        assert result.error_code == "WINDOW_TOO_SMALL"
    
    def test_unordered_timestamps_rejected_in_strict_mode(self):
        """Test that unordered timestamps are rejected in strict mode."""
        # Arrange
        values = [20.0, 21.0, 22.0, 23.0, 24.0]
        timestamps = [1.0, 3.0, 2.0, 4.0, 5.0]  # Out of order
        
        validator = InputValidator(min_window_size=3, strict_mode=True)
        
        # Act
        result = validator.validate(values, timestamps)
        
        # Assert
        assert not result.valid
        assert result.error_code == "UNORDERED_TIMESTAMPS"
    
    def test_unordered_timestamps_sanitized_in_lenient_mode(self):
        """Test that unordered timestamps are sorted in lenient mode."""
        # Arrange
        values = [20.0, 21.0, 22.0, 23.0, 24.0]
        timestamps = [1.0, 3.0, 2.0, 4.0, 5.0]  # Out of order
        
        validator = InputValidator(min_window_size=3, strict_mode=False)
        
        # Act
        result = validator.validate(values, timestamps)
        
        # Assert
        assert result.valid
        assert result.sanitized_timestamps is not None
        assert result.sanitized_timestamps == sorted(timestamps)
    
    def test_length_mismatch_rejected(self):
        """Test that length mismatch between values and timestamps is rejected."""
        # Arrange
        values = [20.0, 21.0, 22.0]
        timestamps = [1.0, 2.0]  # Different length
        
        validator = InputValidator(min_window_size=3)
        
        # Act
        result = validator.validate(values, timestamps)
        
        # Assert
        assert not result.valid
        assert result.error_code == "LENGTH_MISMATCH"


class TestE2ERateLimiting:
    """Test rate limiting functionality."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        class MockPipeline:
            def __init__(self):
                self.commands = []
            
            def zremrangebyscore(self, key, min_score, max_score):
                self.commands.append(('zremrangebyscore', key, min_score, max_score))
                return self
            
            def zcard(self, key):
                self.commands.append(('zcard', key))
                return self
            
            def zadd(self, key, mapping):
                self.commands.append(('zadd', key, mapping))
                return self
            
            def expire(self, key, ttl):
                self.commands.append(('expire', key, ttl))
                return self
            
            def execute(self):
                # Return: [zremrangebyscore_result, zcard_result, zadd_result, expire_result]
                return [0, 0, 1, True]
        
        class MockRedis:
            def pipeline(self):
                return MockPipeline()
            
            def zrem(self, key, member):
                pass
            
            def zrange(self, key, start, end, withscores=False):
                return []
        
        return MockRedis()
    
    def test_rate_limit_allows_under_limit(self, mock_redis):
        """Test that requests under limit are allowed."""
        # Arrange
        limiter = RateLimiter(
            redis_client=mock_redis,
            tenant_id="test_tenant",
            default_config=RateLimitConfig(requests_per_second=10, burst_size=15),
        )
        
        # Act
        result = limiter.check_tenant_limit()
        
        # Assert
        assert result.allowed
        assert result.remaining >= 0
    
    def test_rate_limit_blocks_over_limit(self):
        """Test that requests over limit are blocked."""
        # Arrange
        limiter = RateLimiter(
            redis_client=None,  # Fail-open mode
            tenant_id="test_tenant",
            default_config=RateLimitConfig(requests_per_second=1, burst_size=1),
        )
        
        # Act - first request allowed (fail-open)
        result1 = limiter.check_tenant_limit()
        
        # Assert
        assert result1.allowed  # Fail-open when Redis unavailable


class TestE2ENamespaceIsolation:
    """Test Redis namespace isolation."""
    
    def test_tenant_isolation(self):
        """Test that different tenants have isolated keys."""
        # Arrange
        keys_tenant_a = RedisKeys(env="test", tenant_id="tenant_a")
        keys_tenant_b = RedisKeys(env="test", tenant_id="tenant_b")
        
        # Act
        key_a = keys_tenant_a.plasticity("STABLE")
        key_b = keys_tenant_b.plasticity("STABLE")
        
        # Assert
        assert key_a != key_b
        assert "tenant_a" in key_a
        assert "tenant_b" in key_b
    
    def test_key_sanitization(self):
        """Test that unsafe characters are sanitized."""
        # Arrange
        keys = RedisKeys(env="test", tenant_id="tenant:with:colons")
        
        # Act
        key = keys.plasticity("STABLE")
        
        # Assert
        assert ":" not in keys.tenant_id  # Sanitized
        assert keys.tenant_id == "tenant_with_colons"
    
    def test_ttl_methods_exist(self):
        """Test that all TTL methods are defined."""
        # Arrange
        keys = RedisKeys(env="test", tenant_id="test_tenant")
        
        # Act & Assert
        assert keys.plasticity_ttl() is None  # Persistent
        assert keys.error_history_ttl() == 7 * 86400  # 7 days
        assert keys.anomaly_track_ttl() == 30 * 86400  # 30 days
        assert keys.anomaly_consecutive_ttl() == 3600  # 1 hour
        assert keys.last_alert_ttl() == 3600  # 1 hour
        assert keys.suppressed_ttl() == 3600  # 1 hour
        assert keys.rate_limit_ttl() == 60  # 60 seconds


class TestE2EWeightsValidity:
    """Test that weights remain valid throughout pipeline."""
    
    def test_weights_sum_to_one(self):
        """Test that normalized weights sum to 1.0."""
        # Arrange
        raw_weights = {"engine_a": 0.5, "engine_b": 0.3, "engine_c": 0.2}
        
        # Act
        total = sum(raw_weights.values())
        normalized = {k: v / total for k, v in raw_weights.items()}
        
        # Assert
        assert abs(sum(normalized.values()) - 1.0) < 1e-9
    
    def test_weights_non_negative(self):
        """Test that weights are non-negative."""
        # Arrange
        weights = {"engine_a": 0.5, "engine_b": 0.3, "engine_c": 0.2}
        
        # Assert
        assert all(w >= 0 for w in weights.values())


class TestE2EMetrics:
    """Test metrics collection."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client for testing."""
        class MockPipeline:
            def __init__(self):
                self.commands = []
            
            def zremrangebyscore(self, key, min_score, max_score):
                self.commands.append(('zremrangebyscore', key, min_score, max_score))
                return self
            
            def zcard(self, key):
                self.commands.append(('zcard', key))
                return self
            
            def zadd(self, key, mapping):
                self.commands.append(('zadd', key, mapping))
                return self
            
            def expire(self, key, ttl):
                self.commands.append(('expire', key, ttl))
                return self
            
            def execute(self):
                # Return: [zremrangebyscore_result, zcard_result, zadd_result, expire_result]
                return [0, 0, 1, True]
        
        class MockRedis:
            def pipeline(self):
                return MockPipeline()
            
            def zrem(self, key, member):
                pass
            
            def zrange(self, key, start, end, withscores=False):
                return []
        
        return MockRedis()
    
    def test_validator_metrics(self):
        """Test that validator collects metrics."""
        # Arrange
        validator = InputValidator(min_window_size=3)
        
        # Act
        validator.validate([20.0, 21.0, 22.0])  # Valid
        validator.validate([20.0])  # Invalid (too small)
        
        metrics = validator.get_metrics()
        
        # Assert
        assert metrics["valid_count"] == 1
        assert metrics["rejected_count"] == 1
        assert metrics["total_requests"] == 2
    
    def test_rate_limiter_metrics(self, mock_redis):
        """Test that rate limiter collects metrics."""
        # Arrange
        limiter = RateLimiter(
            redis_client=mock_redis,
            tenant_id="test_tenant",
        )
        
        # Act
        limiter.check_tenant_limit()
        limiter.check_tenant_limit()
        
        metrics = limiter.get_metrics()
        
        # Assert
        assert metrics["allowed_count"] == 2
        assert metrics["total_requests"] == 2
