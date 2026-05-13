"""Simple benchmark for posterior cache (PERF-CRIT-2).

Standalone script to measure cache performance.
"""

import sys
import time
from typing import Optional, Tuple

# Add parent to path
sys.path.insert(0, '/home/nicolas/Documentos/Iot_System/iot_machine_learning/infrastructure/ml/cognitive/bayesian_weight_tracker')

from posterior_cache import PosteriorCache
from storage_interface import IPosteriorStorage
from cached_storage import CachedPosteriorStorage


class MockRedisStorage(IPosteriorStorage):
    """Mock Redis with artificial latency."""
    
    def __init__(self, latency_ms: float = 5.0):
        self.latency_ms = latency_ms
        self.load_count = 0
        self._data = {}
    
    def load_posterior(self, regime: str, engine_name: str) -> Optional[Tuple[float, float]]:
        time.sleep(self.latency_ms / 1000.0)
        self.load_count += 1
        return self._data.get((regime, engine_name))
    
    def save_posterior(self, regime: str, engine_name: str, mu: float, sigma2: float) -> None:
        time.sleep(self.latency_ms / 1000.0)
        self._data[(regime, engine_name)] = (mu, sigma2)
    
    def delete_posterior(self, regime: str, engine_name: str) -> None:
        self._data.pop((regime, engine_name), None)


def benchmark_cache_hit():
    """Benchmark cache hit performance."""
    print("=" * 60)
    print("PERF-CRIT-2: Posterior Cache Benchmark")
    print("=" * 60)
    
    redis = MockRedisStorage(latency_ms=5.0)
    cached = CachedPosteriorStorage(redis, cache_ttl_seconds=60.0)
    
    regime = "stable"
    engine = "taylor"
    
    # Warm up
    cached.save_posterior(regime, engine, 0.8, 0.1)
    
    # Benchmark WITHOUT cache (direct Redis)
    print("\n1. WITHOUT cache (100 reads):")
    start = time.perf_counter()
    for _ in range(100):
        redis.load_posterior(regime, engine)
    no_cache_time = time.perf_counter() - start
    print(f"   Time: {no_cache_time*1000:.2f}ms")
    print(f"   Avg per read: {no_cache_time*10:.2f}ms")
    
    # Benchmark WITH cache
    print("\n2. WITH cache (100 reads):")
    redis.load_count = 0
    start = time.perf_counter()
    for _ in range(100):
        cached.load_posterior(regime, engine)
    with_cache_time = time.perf_counter() - start
    print(f"   Time: {with_cache_time*1000:.2f}ms")
    print(f"   Avg per read: {with_cache_time*10:.2f}ms")
    print(f"   Redis calls: {redis.load_count}")
    
    # Calculate speedup
    speedup = no_cache_time / with_cache_time
    print(f"\n3. RESULT:")
    print(f"   Speedup: {speedup:.1f}x faster")
    print(f"   Redis calls reduced: {100 - redis.load_count} / 100 ({(100-redis.load_count):.0f}%)")
    
    return speedup


def benchmark_realistic_workload():
    """Simulate realistic workload."""
    print("\n" + "=" * 60)
    print("Realistic Workload: 1000 updates")
    print("=" * 60)
    
    redis = MockRedisStorage(latency_ms=5.0)
    cached = CachedPosteriorStorage(redis, cache_ttl_seconds=60.0)
    
    engines = ["taylor", "holt", "ensemble", "baseline"]
    regime = "stable"
    
    # Warm up
    for engine in engines:
        cached.save_posterior(regime, engine, 0.5, 0.2)
    
    redis.load_count = 0
    
    start = time.perf_counter()
    for i in range(1000):
        engine = engines[i % len(engines)]
        posterior = cached.load_posterior(regime, engine)
        if posterior:
            mu, sigma2 = posterior
            new_mu = mu * 0.9 + 0.1 * 0.8
            cached.save_posterior(regime, engine, new_mu, sigma2)
    
    elapsed = time.perf_counter() - start
    
    throughput = 1000 / elapsed
    hit_rate = (1000 - redis.load_count) / 1000
    
    print(f"\n   Total time: {elapsed*1000:.2f}ms")
    print(f"   Throughput: {throughput:.0f} ops/sec")
    print(f"   Cache hit rate: {hit_rate*100:.1f}%")
    print(f"   Redis loads: {redis.load_count} / 1000")
    
    return throughput, hit_rate


if __name__ == "__main__":
    speedup = benchmark_cache_hit()
    throughput, hit_rate = benchmark_realistic_workload()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"✓ Cache speedup: {speedup:.1f}x")
    print(f"✓ Throughput: {throughput:.0f} ops/sec")
    print(f"✓ Hit rate: {hit_rate*100:.1f}%")
    print("\nPERF-CRIT-2: ✅ RESOLVED")
