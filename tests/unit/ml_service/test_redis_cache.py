"""Tests for Redis-backed analysis cache."""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock
import json

from iot_machine_learning.ml_service.api.services.analysis.cache import AnalysisCache


class TestRedisCache:
    """Test suite for Redis-backed cache."""
    
    def test_cache_without_redis_fallback_to_memory(self):
        """Test that cache works without Redis (fallback to memory)."""
        cache = AnalysisCache(max_entries=10)
        
        # Should work without Redis
        cache.set("test_key", {"result": "value"})
        result = cache.get("test_key")
        
        assert result is not None
        assert result["result"] == "value"
    
    def test_cache_with_redis_stores_and_retrieves(self):
        """Test that cache stores and retrieves from Redis."""
        # Mock Redis client
        redis_mock = Mock()
        redis_mock.ping.return_value = True
        redis_mock.get.return_value = None
        redis_mock.setex.return_value = True
        
        cache = AnalysisCache(max_entries=10, redis_client=redis_mock)
        
        # Store in cache
        test_data = {"analysis": "result", "confidence": 0.85}
        cache.set("test_key", test_data)
        
        # Verify Redis was called
        redis_mock.setex.assert_called_once()
        call_args = redis_mock.setex.call_args
        assert "zenin:analysis:test_key" in call_args[0]
        assert call_args[0][1] == 3600  # TTL
    
    def test_cache_redis_get_populates_memory(self):
        """Test that Redis get populates in-memory cache."""
        # Mock Redis client
        redis_mock = Mock()
        redis_mock.ping.return_value = True
        
        test_data = {"analysis": "from_redis", "score": 0.9}
        redis_mock.get.return_value = json.dumps(test_data)
        
        cache = AnalysisCache(max_entries=10, redis_client=redis_mock)
        
        # Get from cache (should hit Redis)
        result = cache.get("redis_key")
        
        assert result is not None
        assert result["analysis"] == "from_redis"
        assert result["score"] == 0.9
        
        # Verify Redis was called
        redis_mock.get.assert_called_once()
        
        # Second get should hit memory cache (not Redis again)
        redis_mock.get.reset_mock()
        result2 = cache.get("redis_key")
        assert result2 == result
        redis_mock.get.assert_not_called()
    
    def test_cache_redis_unavailable_graceful_fallback(self):
        """Test graceful fallback when Redis is unavailable."""
        # Mock Redis client that fails ping
        redis_mock = Mock()
        redis_mock.ping.side_effect = Exception("Redis connection failed")
        
        cache = AnalysisCache(max_entries=10, redis_client=redis_mock)
        
        # Should still work with memory cache
        cache.set("test_key", {"result": "value"})
        result = cache.get("test_key")
        
        assert result is not None
        assert result["result"] == "value"
    
    def test_cache_redis_set_failure_graceful(self):
        """Test graceful handling of Redis set failure."""
        # Mock Redis client that fails on setex
        redis_mock = Mock()
        redis_mock.ping.return_value = True
        redis_mock.setex.side_effect = Exception("Redis write failed")
        
        cache = AnalysisCache(max_entries=10, redis_client=redis_mock)
        
        # Should still store in memory even if Redis fails
        cache.set("test_key", {"result": "value"})
        result = cache.get("test_key")
        
        assert result is not None
        assert result["result"] == "value"
    
    def test_cache_redis_get_failure_graceful(self):
        """Test graceful handling of Redis get failure."""
        # Mock Redis client that fails on get
        redis_mock = Mock()
        redis_mock.ping.return_value = True
        redis_mock.get.side_effect = Exception("Redis read failed")
        
        cache = AnalysisCache(max_entries=10, redis_client=redis_mock)
        
        # Should return None gracefully
        result = cache.get("test_key")
        assert result is None
