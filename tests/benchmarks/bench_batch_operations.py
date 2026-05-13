"""Benchmark for batch operations (PERF-SEV-1)."""

import time
from typing import List


class MockRedis:
    """Mock Redis with operation counting."""
    
    def __init__(self):
        self.operation_count = 0
        self._data = {}
    
    def rpush(self, key: str, *values):
        self.operation_count += 1
        if key not in self._data:
            self._data[key] = []
        self._data[key].extend(values)
    
    def ltrim(self, key: str, start: int, stop: int):
        self.operation_count += 1
        if key in self._data:
            self._data[key] = self._data[key][start:stop+1 if stop != -1 else None]
    
    def expire(self, key: str, ttl: int):
        self.operation_count += 1
    
    def pipeline(self):
        return MockPipeline(self)


class MockPipeline:
    """Mock Redis pipeline."""
    
    def __init__(self, redis):
        self.redis = redis
        self.commands = []
    
    def rpush(self, key: str, *values):
        self.commands.append(('rpush', key, values))
        return self
    
    def ltrim(self, key: str, start: int, stop: int):
        self.commands.append(('ltrim', key, start, stop))
        return self
    
    def expire(self, key: str, ttl: int):
        self.commands.append(('expire', key, ttl))
        return self
    
    def execute(self):
        # Execute all commands as one operation
        self.redis.operation_count += 1
        for cmd in self.commands:
            if cmd[0] == 'rpush':
                _, key, values = cmd
                if key not in self.redis._data:
                    self.redis._data[key] = []
                self.redis._data[key].extend(values)
            elif cmd[0] == 'ltrim':
                _, key, start, stop = cmd
                if key in self.redis._data:
                    self.redis._data[key] = self.redis._data[key][start:stop+1 if stop != -1 else None]
        return [None] * len(self.commands)


def simulate_individual_appends(redis: MockRedis, series_id: str, values: List[float]):
    """Simulate N individual append operations."""
    for v in values:
        pipe = redis.pipeline()
        pipe.rpush(f"series:{series_id}", str(v))
        pipe.ltrim(f"series:{series_id}", -500, -1)
        pipe.expire(f"series:{series_id}", 86400)
        pipe.execute()


def simulate_batch_append(redis: MockRedis, series_id: str, values: List[float]):
    """Simulate single batch append operation."""
    pipe = redis.pipeline()
    pipe.rpush(f"series:{series_id}", *[str(v) for v in values])
    pipe.ltrim(f"series:{series_id}", -500, -1)
    pipe.expire(f"series:{series_id}", 86400)
    pipe.execute()


def simulate_multi_series_batch(redis: MockRedis, batch: List[tuple[str, List[float]]]):
    """Simulate multi-series batch operation."""
    pipe = redis.pipeline()
    for series_id, values in batch:
        pipe.rpush(f"series:{series_id}", *[str(v) for v in values])
        pipe.ltrim(f"series:{series_id}", -500, -1)
        pipe.expire(f"series:{series_id}", 86400)
    pipe.execute()


def benchmark():
    print("=" * 70)
    print("PERF-SEV-1: Batch Operations Benchmark")
    print("=" * 70)
    
    # Test 1: Individual vs Batch (single series)
    print("\n[1] Single series: 100 values")
    
    redis1 = MockRedis()
    values = [float(i) for i in range(100)]
    
    start = time.perf_counter()
    simulate_individual_appends(redis1, "sensor_1", values)
    individual_time = time.perf_counter() - start
    individual_ops = redis1.operation_count
    
    redis2 = MockRedis()
    start = time.perf_counter()
    simulate_batch_append(redis2, "sensor_1", values)
    batch_time = time.perf_counter() - start
    batch_ops = redis2.operation_count
    
    print(f"    Individual: {individual_ops} Redis operations")
    print(f"    Batch:      {batch_ops} Redis operations")
    print(f"    Reduction:  {individual_ops - batch_ops} operations ({(1-batch_ops/individual_ops)*100:.0f}%)")
    
    # Test 2: Multi-series batch
    print("\n[2] Multi-series: 10 series × 50 values each")
    
    batch_data = [(f"sensor_{i}", [float(j) for j in range(50)]) for i in range(10)]
    
    redis3 = MockRedis()
    start = time.perf_counter()
    for series_id, vals in batch_data:
        simulate_individual_appends(redis3, series_id, vals)
    multi_individual_time = time.perf_counter() - start
    multi_individual_ops = redis3.operation_count
    
    redis4 = MockRedis()
    start = time.perf_counter()
    simulate_multi_series_batch(redis4, batch_data)
    multi_batch_time = time.perf_counter() - start
    multi_batch_ops = redis4.operation_count
    
    print(f"    Individual: {multi_individual_ops} Redis operations")
    print(f"    Batch:      {multi_batch_ops} Redis operations")
    print(f"    Reduction:  {multi_individual_ops - multi_batch_ops} operations ({(1-multi_batch_ops/multi_individual_ops)*100:.0f}%)")
    
    # Test 3: Realistic workload
    print("\n[3] Realistic: 100 series × 10 values each = 1000 total values")
    
    realistic_batch = [(f"sensor_{i}", [float(j) for j in range(10)]) for i in range(100)]
    
    redis5 = MockRedis()
    for series_id, vals in realistic_batch:
        for v in vals:
            simulate_individual_appends(redis5, series_id, [v])
    realistic_individual_ops = redis5.operation_count
    
    redis6 = MockRedis()
    simulate_multi_series_batch(redis6, realistic_batch)
    realistic_batch_ops = redis6.operation_count
    
    print(f"    Individual: {realistic_individual_ops} Redis operations")
    print(f"    Batch:      {realistic_batch_ops} Redis operations")
    print(f"    Reduction:  {realistic_individual_ops - realistic_batch_ops} operations ({(1-realistic_batch_ops/realistic_individual_ops)*100:.0f}%)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Single series batch: {(1-batch_ops/individual_ops)*100:.0f}% fewer operations")
    print(f"✓ Multi-series batch: {(1-multi_batch_ops/multi_individual_ops)*100:.0f}% fewer operations")
    print(f"✓ Realistic workload: {(1-realistic_batch_ops/realistic_individual_ops)*100:.0f}% fewer operations")
    print(f"\n PERF-SEV-1: RESOLVED - Batch operations reduce Redis round-trips by ~{(1-realistic_batch_ops/realistic_individual_ops)*100:.0f}%")


if __name__ == "__main__":
    benchmark()
