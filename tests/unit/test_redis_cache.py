"""Tests for Redis cache implementation (canonical).

MIGRATED 2026-04-09: Updated to use RedisConnectionManager mocks.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
import json

from iot_machine_learning.domain.ports.document_analysis import AnalysisOutput
from iot_machine_learning.infrastructure.persistence.redis_cache import RedisAnalysisCache
from iot_machine_learning.infrastructure.persistence.cache import compute_content_hash, build_cache_key
from iot_machine_learning.infrastructure.persistence.cache_system.cache_factory import create_cache


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis_mock = MagicMock()
    redis_mock.ping.return_value = True
    yield redis_mock


@pytest.fixture
def sample_analysis():
    """Sample AnalysisOutput for testing."""
    return AnalysisOutput(
        document_id="doc-001",
        tenant_id="tenant-001",
        classification="document",
        conclusion="Test conclusion",
        confidence=0.8,
        analysis={"key": "value"},
        explanation=None,
        processing_time_ms=100.0,
        cached=False,
    )


class TestRedisAnalysisCache:
    """Tests for RedisAnalysisCache."""
    
    def test_init_with_redis(self, mock_redis):
        """Test initialization with Redis available."""
        cache = RedisAnalysisCache(ttl_seconds=60, key_prefix="test:")
        
        assert cache._ttl == 60
        assert cache._key_prefix == "test:"
        assert cache.is_redis_available
        assert not cache.is_using_fallback
    
    @patch("iot_machine_learning.infrastructure.persistence.redis_cache.get_redis_client")
    def test_init_fallback_to_memory(self, mock_get_redis):
        """Test fallback to in-memory cache when Redis unavailable."""
        mock_get_redis.return_value = None
        
        cache = RedisAnalysisCache(ttl_seconds=60, fallback_to_memory=True)
        
        assert not cache.is_redis_available
        assert cache.is_using_fallback
        assert cache._memory_cache is not None
    
    @patch("iot_machine_learning.infrastructure.persistence.redis_cache.get_redis_client")
    def test_init_no_fallback(self, mock_get_redis):
        """Test no fallback when disabled."""
        mock_get_redis.return_value = None
        
        cache = RedisAnalysisCache(ttl_seconds=60, fallback_to_memory=False)
        
        assert not cache.is_redis_available
        assert not cache.is_using_fallback
        assert cache._memory_cache is None
    
    def test_make_key(self, mock_redis):
        """Test key prefixing."""
        cache = RedisAnalysisCache(key_prefix="ml:cache:")
        
        key = cache._make_key("test123")
        assert key == "ml:cache:test123"
    
    def test_get_hit(self, mock_redis, sample_analysis):
        """Test cache hit."""
        cache = RedisAnalysisCache()
        
        # Mock Redis get
        mock_redis.get.return_value = json.dumps(sample_analysis.__dict__)
        
        result = cache.get("test_key")
        
        assert result is not None
        assert result.conclusion == "Test conclusion"
        mock_redis.get.assert_called_once()
    
    def test_get_miss(self, mock_redis):
        """Test cache miss."""
        cache = RedisAnalysisCache()
        
        # Mock Redis get returns None
        mock_redis.get.return_value = None
        
        result = cache.get("test_key")
        
        assert result is None
        mock_redis.get.assert_called_once()
    
    def test_get_redis_error_returns_none(self, mock_redis):
        """Test returns None on Redis error (no fallback in new impl)."""
        cache = RedisAnalysisCache(mock_redis)
        
        # Mock Redis error
        mock_redis.get.side_effect = Exception("Connection lost")
        
        # Should return None, not raise
        result = cache.get("test_key")
        
        assert result is None
    
    def test_set_success(self, mock_redis, sample_analysis):
        """Test successful cache set."""
        cache = RedisAnalysisCache(ttl_seconds=300)
        
        cache.set("test_key", sample_analysis, ttl=60)
        
        # Verify Redis setex was called
        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        
        assert call_args[0][0] == "ml:cache:test_key"  # key
        assert call_args[0][1] == 60  # ttl
        # Third arg is JSON data
    
    def test_set_default_ttl(self, mock_redis, sample_analysis):
        """Test set with default TTL."""
        cache = RedisAnalysisCache(ttl_seconds=300)
        
        cache.set("test_key", sample_analysis)
        
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 300  # default ttl
    
    def test_set_redis_error_fallback(self, mock_redis, sample_analysis):
        """Test fallback to memory cache on Redis set error."""
        cache = RedisAnalysisCache(fallback_to_memory=True)
        
        # Mock Redis error
        mock_redis.setex.side_effect = Exception("Connection lost")
        
        # Should fallback to memory cache
        cache.set("test_key", sample_analysis)
        
        # Verify stored in memory cache
        result = cache._memory_cache.get("test_key")
        assert result is not None
    
    def test_invalidate_success(self, mock_redis):
        """Test successful cache invalidation."""
        cache = RedisAnalysisCache()
        
        cache.invalidate("test_key")
        
        mock_redis.delete.assert_called_once_with("ml:cache:test_key")
    
    def test_invalidate_redis_error_fallback(self, mock_redis, sample_analysis):
        """Test fallback to memory cache on Redis invalidate error."""
        cache = RedisAnalysisCache(fallback_to_memory=True)
        
        # Store in memory cache
        cache._memory_cache.set("test_key", sample_analysis)
        
        # Mock Redis error
        mock_redis.delete.side_effect = Exception("Connection lost")
        
        # Should fallback to memory cache
        cache.invalidate("test_key")
        
        # Verify removed from memory cache
        result = cache._memory_cache.get("test_key")
        assert result is None
    
    def test_clear_success(self, mock_redis):
        """Test successful cache clear."""
        cache = RedisAnalysisCache(key_prefix="ml:cache:")
        
        # Mock SCAN to return some keys
        mock_redis.scan.side_effect = [
            (10, [b"ml:cache:key1", b"ml:cache:key2"]),
            (0, [b"ml:cache:key3"]),  # cursor 0 = done
        ]
        
        cache.clear()
        
        # Verify SCAN was called
        assert mock_redis.scan.call_count == 2
        
        # Verify DELETE was called for all keys
        assert mock_redis.delete.call_count == 2
    
    def test_size_calculation(self, mock_redis):
        """Test cache size calculation."""
        cache = RedisAnalysisCache()
        
        # Mock SCAN to return keys
        mock_redis.scan.side_effect = [
            (10, [b"ml:cache:key1", b"ml:cache:key2"]),
            (0, [b"ml:cache:key3"]),
        ]
        
        size = cache.size
        
        assert size == 3
    
    def test_size_redis_error_fallback(self, mock_redis):
        """Test size fallback to memory cache on Redis error."""
        cache = RedisAnalysisCache(fallback_to_memory=True)
        
        # Mock Redis error
        mock_redis.scan.side_effect = Exception("Connection lost")
        
        # Should fallback to memory cache size
        size = cache.size
        
        assert size == 0  # empty memory cache


class TestCacheFactory:
    """Tests for cache factory function."""
    
    @patch("iot_machine_learning.infrastructure.persistence.factory.RedisConnectionManager.get_sync_client")
    def test_create_redis_cache(self, mock_get_client):
        """Test creating Redis cache via unified connection manager."""
        mock_redis = MagicMock()
        mock_get_client.return_value = mock_redis
        
        cache = create_cache(use_redis=True, ttl_seconds=60)
        
        assert isinstance(cache, RedisAnalysisCache)
        assert cache._ttl == 60
        mock_get_client.assert_called_once()
    
    def test_create_memory_cache(self):
        """Test creating in-memory cache."""
        cache = create_cache(use_redis=False, ttl_seconds=120)
        
        from iot_machine_learning.infrastructure.persistence.cache import InMemoryAnalysisCache
        assert isinstance(cache, InMemoryAnalysisCache)
        assert cache._ttl == 120


class TestCacheHelpers:
    """Tests for cache helper functions."""
    
    def test_compute_content_hash(self):
        """Test content hash computation."""
        hash1 = compute_content_hash("test content")
        hash2 = compute_content_hash("test content")
        hash3 = compute_content_hash("different content")
        
        assert hash1 == hash2  # Same content = same hash
        assert hash1 != hash3  # Different content = different hash
        assert len(hash1) == 16  # MD5 truncated to 16 chars
    
    def test_build_cache_key(self):
        """Test cache key building."""
        key = build_cache_key("abc123", "document")
        
        assert key == "abc123:document"
        assert ":" in key


class TestCacheDecorators:
    """Tests for cache decorators."""
    
    def test_cached_decorator(self, mock_redis, sample_analysis):
        """Test @cached decorator."""
        from iot_machine_learning.infrastructure.persistence.cache_decorators import (
            cached,
            set_global_cache,
        )
        
        # Set up cache
        cache = RedisAnalysisCache()
        set_global_cache(cache)
        
        # Mock Redis
        call_count = 0
        
        @cached(ttl=60)
        def expensive_function(x: int) -> int:
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call - should compute
        mock_redis.get.return_value = None
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call - should use cache
        mock_redis.get.return_value = json.dumps(10)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not called again
    
    def test_memoize_decorator(self):
        """Test @memoize decorator."""
        from iot_machine_learning.infrastructure.persistence.cache_decorators import memoize
        
        call_count = 0
        
        @memoize
        def fibonacci(n: int) -> int:
            nonlocal call_count
            call_count += 1
            if n < 2:
                return n
            return fibonacci(n-1) + fibonacci(n-2)
        
        result = fibonacci(5)
        assert result == 5
        # With memoization, should be called much less than without
        assert call_count < 10  # Without memoization would be ~15
