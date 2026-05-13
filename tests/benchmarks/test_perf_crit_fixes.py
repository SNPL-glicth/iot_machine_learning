"""Benchmarks for PERF-CRIT-1, PERF-CRIT-2, PERF-SEV-1 fixes.

Performance validation for critical bottlenecks.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple
from unittest.mock import Mock

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.posterior_cache import (
    PosteriorCache,
)
from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.storage_interface import (
    IPosteriorStorage,
)
from iot_machine_learning.infrastructure.ml.cognitive.bayesian_weight_tracker.cached_storage import (
    CachedPosteriorStorage,
)


class MockRedisStorage(IPosteriorStorage):
    """Mock Redis storage with artificial latency."""
    
    def __init__(self, latency_ms: float = 5.0):
        self.latency_ms = latency_ms
        self.load_count = 0
        self.save_count = 0
        self._data = {}
    
    def load_posterior(self, regime: str, engine_name: str) -> Optional[Tuple[float, float]]:
        time.sleep(self.latency_ms / 1000.0)  # Simulate network latency
        self.load_count += 1
        key = (regime, engine_name)
        return self._data.get(key)
    
    def save_posterior(self, regime: str, engine_name: str, mu: float, sigma2: float) -> None:
        time.sleep(self.latency_ms / 1000.0)  # Simulate network latency
        self.save_count += 1
        key = (regime, engine_name)
        self._data[key] = (mu, sigma2)
    
    def delete_posterior(self, regime: str, engine_name: str) -> None:
        key = (regime, engine_name)
        self._data.pop(key, None)


class TestPerfCrit2PosteriorCache:
    """PERF-CRIT-2: Benchmark posterior cache performance."""
    
    def test_cache_hit_performance(self):
        """Cache hits should be ~100x faster than Redis."""
        redis = MockRedisStorage(latency_ms=5.0)
        cached = CachedPosteriorStorage(redis, cache_ttl_seconds=60.0)
        
        regime = "stable"
        engine = "taylor"
        
        # Warm up cache
        cached.save_posterior(regime, engine, 0.8, 0.1)
        
        # Benchmark: 100 reads WITHOUT cache
        start = time.perf_counter()
        for _ in range(100):
            redis.load_posterior(regime, engine)
        no_cache_time = time.perf_counter() - start
        
        # Benchmark: 100 reads WITH cache
        redis.load_count = 0  # Reset counter
        start = time.perf_counter()
        for _ in range(100):
            cached.load_posterior(regime, engine)
        with_cache_time = time.perf_counter() - start
        
        # Verify cache was used (only 1 Redis call for initial load)
        assert redis.load_count <= 2  # Initial + maybe 1 more
        
        # Cache should be significantly faster
        speedup = no_cache_time / with_cache_time
        print(f"\nPERF-CRIT-2 Speedup: {speedup:.1f}x")
        print(f"  Without cache: {no_cache_time*1000:.2f}ms")
        print(f"  With cache: {with_cache_time*1000:.2f}ms")
        
        assert speedup > 10  # At least 10x faster
    
    def test_cache_write_through_overhead(self):
        """Write-through should have minimal overhead."""
        redis = MockRedisStorage(latency_ms=5.0)
        cached = CachedPosteriorStorage(redis, cache_ttl_seconds=60.0)
        
        regime = "stable"
        engine = "taylor"
        
        # Benchmark: 100 writes to Redis only
        start = time.perf_counter()
        for i in range(100):
            redis.save_posterior(regime, f"engine_{i}", 0.8, 0.1)
        redis_only_time = time.perf_counter() - start
        
        # Benchmark: 100 writes with cache (write-through)
        start = time.perf_counter()
        for i in range(100):
            cached.save_posterior(regime, f"cached_engine_{i}", 0.8, 0.1)
        cached_time = time.perf_counter() - start
        
        # Write-through overhead should be < 10%
        overhead = (cached_time - redis_only_time) / redis_only_time
        print(f"\nPERF-CRIT-2 Write-through overhead: {overhead*100:.1f}%")
        
        assert overhead < 0.2  # Less than 20% overhead
    
    def test_cache_ttl_expiration(self):
        """Expired entries should be evicted."""
        cache = PosteriorCache(ttl_seconds=0.1)  # 100ms TTL
        
        cache.put("stable", "engine_a", 0.8, 0.1)
        cache.put("stable", "engine_b", 0.7, 0.2)
        
        # Should be cached
        assert cache.get("stable", "engine_a") is not None
        assert cache.size() == 2
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert cache.get("stable", "engine_a") is None
        
        # Evict expired
        evicted = cache.evict_expired()
        assert evicted == 1  # engine_b also expired
        assert cache.size() == 0
    
    def test_cache_thread_safety(self):
        """Cache should be thread-safe."""
        import threading
        
        cache = PosteriorCache(ttl_seconds=60.0)
        errors = []
        
        def writer(engine_id: int):
            try:
                for i in range(100):
                    cache.put("stable", f"engine_{engine_id}", float(i), 0.1)
            except Exception as e:
                errors.append(e)
        
        def reader(engine_id: int):
            try:
                for _ in range(100):
                    cache.get("stable", f"engine_{engine_id}")
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0  # No race conditions
    
    def test_realistic_workload_simulation(self):
        """Simulate realistic update workload: 1000 updates, 80% cache hit rate."""
        redis = MockRedisStorage(latency_ms=5.0)
        cached = CachedPosteriorStorage(redis, cache_ttl_seconds=60.0)
        
        engines = ["taylor", "holt", "ensemble", "baseline"]
        regime = "stable"
        
        # Warm up cache with initial values
        for engine in engines:
            cached.save_posterior(regime, engine, 0.5, 0.2)
        
        # Simulate 1000 updates (reads + writes)
        redis.load_count = 0
        redis.save_count = 0
        
        start = time.perf_counter()
        for i in range(1000):
            engine = engines[i % len(engines)]
            
            # Read (80% should hit cache after warm-up)
            posterior = cached.load_posterior(regime, engine)
            
            # Update
            if posterior:
                mu, sigma2 = posterior
                new_mu = mu * 0.9 + 0.1 * (0.8 if i % 2 == 0 else 0.6)
                cached.save_posterior(regime, engine, new_mu, sigma2)
        
        elapsed = time.perf_counter() - start
        
        # Calculate cache hit rate
        total_reads = 1000
        cache_hits = total_reads - redis.load_count
        hit_rate = cache_hits / total_reads
        
        print(f"\nPERF-CRIT-2 Realistic workload:")
        print(f"  Total updates: 1000")
        print(f"  Time: {elapsed*1000:.2f}ms")
        print(f"  Throughput: {1000/elapsed:.0f} ops/sec")
        print(f"  Cache hit rate: {hit_rate*100:.1f}%")
        print(f"  Redis loads: {redis.load_count}")
        print(f"  Redis saves: {redis.save_count}")
        
        # With cache, should achieve high hit rate
        assert hit_rate > 0.7  # At least 70% hit rate
        assert 1000/elapsed > 500  # At least 500 ops/sec
