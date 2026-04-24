"""Tests for EngineErrorStore (IMP-4a).

Covers:
1. record+get_recent round-trip (memory backend, happy path)
2. max_entries cap enforcement
3. Percentile computation and empty-state contract
4. RMSE over window
5. NaN / Inf / negative error rejection (defensive dropping, no raise)
6. LRU eviction when max_series_engines is reached
7. Redis backend happy path (fake pipeline)
8. Redis failure → transparent memory fallback (no exception)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.error_store import (
    EngineErrorStore,
)


# -- Fake Redis ------------------------------------------------------------


class _FakePipeline:
    def __init__(self, store: Dict[str, List[str]]) -> None:
        self._store = store
        self._ops: List[tuple] = []

    def rpush(self, key: str, value: Any) -> None:
        self._ops.append(("rpush", key, value))

    def ltrim(self, key: str, start: int, end: int) -> None:
        self._ops.append(("ltrim", key, start, end))

    def expire(self, key: str, ttl: int) -> None:
        self._ops.append(("expire", key, ttl))

    def execute(self) -> None:
        for op in self._ops:
            if op[0] == "rpush":
                self._store.setdefault(op[1], []).append(str(op[2]))
            elif op[0] == "ltrim":
                lst = self._store.get(op[1], [])
                self._store[op[1]] = lst[op[2]:] if op[2] < 0 else lst[op[2]:op[3] + 1]


class _FakeRedis:
    def __init__(self) -> None:
        self.store: Dict[str, List[str]] = {}

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self.store)

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        lst = self.store.get(key, [])
        return lst if end == -1 else lst[start:end + 1]


class _BrokenRedis:
    def pipeline(self):  # pragma: no cover — triggered via exception
        raise RuntimeError("redis down")

    def lrange(self, *_args, **_kwargs):  # pragma: no cover
        raise RuntimeError("redis down")


# -- Happy-path memory backend --------------------------------------------


class TestMemoryBackend:
    def test_record_and_get_recent(self) -> None:
        store = EngineErrorStore()
        for e in [1.0, 2.0, 3.0, 4.0]:
            store.record("s1", "eng", e)
        assert store.get_recent("s1", "eng", 10) == [1.0, 2.0, 3.0, 4.0]
        assert store.get_recent("s1", "eng", 2) == [3.0, 4.0]
        assert store.get_recent("s1", "eng", 0) == []

    def test_max_entries_enforced(self) -> None:
        store = EngineErrorStore(max_entries=3)
        for e in [1.0, 2.0, 3.0, 4.0, 5.0]:
            store.record("s1", "eng", e)
        assert store.get_recent("s1", "eng", 10) == [3.0, 4.0, 5.0]

    def test_unknown_series_returns_empty(self) -> None:
        store = EngineErrorStore()
        assert store.get_recent("nope", "eng", 10) == []
        assert store.get_percentile("nope", "eng", 75) == 0.0
        assert store.get_rmse_window("nope", "eng", 10) == 0.0


# -- Percentile + RMSE ----------------------------------------------------


class TestPercentileAndRmse:
    def test_percentile_known_values(self) -> None:
        store = EngineErrorStore()
        for e in [0.0, 1.0, 2.0, 3.0, 4.0]:
            store.record("s", "e", e)
        # p50 = median = 2.0; p75 = 3.0; p100 = 4.0
        assert store.get_percentile("s", "e", 50) == pytest.approx(2.0)
        assert store.get_percentile("s", "e", 75) == pytest.approx(3.0)
        assert store.get_percentile("s", "e", 100) == pytest.approx(4.0)

    def test_percentile_interpolates(self) -> None:
        store = EngineErrorStore()
        for e in [0.0, 10.0]:
            store.record("s", "e", e)
        # 50% between 0 and 10 = 5.0
        assert store.get_percentile("s", "e", 50) == pytest.approx(5.0)

    def test_percentile_out_of_range_raises(self) -> None:
        store = EngineErrorStore()
        store.record("s", "e", 1.0)
        with pytest.raises(ValueError):
            store.get_percentile("s", "e", -1)
        with pytest.raises(ValueError):
            store.get_percentile("s", "e", 101)

    def test_rmse_window(self) -> None:
        store = EngineErrorStore()
        for e in [3.0, 4.0]:  # rmse = sqrt((9+16)/2) = sqrt(12.5)
            store.record("s", "e", e)
        assert store.get_rmse_window("s", "e", 10) == pytest.approx(math.sqrt(12.5))


# -- Defensive input handling --------------------------------------------


class TestInvalidInput:
    def test_nan_inf_and_negative_dropped_silently(self) -> None:
        store = EngineErrorStore()
        store.record("s", "e", float("nan"))
        store.record("s", "e", float("inf"))
        store.record("s", "e", -1.0)
        store.record("s", "e", 2.0)  # only this survives
        assert store.get_recent("s", "e", 10) == [2.0]

    def test_non_numeric_dropped_silently(self) -> None:
        store = EngineErrorStore()
        store.record("s", "e", "not-a-number")  # type: ignore[arg-type]
        assert store.get_recent("s", "e", 10) == []

    def test_invalid_constructor_args(self) -> None:
        with pytest.raises(ValueError):
            EngineErrorStore(max_entries=0)
        with pytest.raises(ValueError):
            EngineErrorStore(ttl_seconds=0)


# -- LRU in-memory eviction ----------------------------------------------


class TestLruEviction:
    def test_lru_evicts_oldest_pair(self) -> None:
        store = EngineErrorStore(max_series_engines=2)
        store.record("a", "e", 1.0)
        store.record("b", "e", 1.0)
        # touch "a" again — now "b" is oldest
        store.record("a", "e", 2.0)
        store.record("c", "e", 1.0)  # this evicts "b"
        assert store.get_recent("a", "e", 10) == [1.0, 2.0]
        assert store.get_recent("c", "e", 10) == [1.0]
        assert store.get_recent("b", "e", 10) == []


# -- Redis backend --------------------------------------------------------


class TestRedisBackend:
    def test_redis_record_and_read_roundtrip(self) -> None:
        fake = _FakeRedis()
        store = EngineErrorStore(redis_client=fake, key_prefix="test_es", ttl_seconds=60)
        for e in [1.0, 2.0, 3.0]:
            store.record("s1", "eng", e)
        assert store.get_recent("s1", "eng", 10) == [1.0, 2.0, 3.0]
        assert "test_es:s1:eng" in fake.store

    def test_redis_failure_falls_back_to_memory(self) -> None:
        store = EngineErrorStore(redis_client=_BrokenRedis())
        # record does not raise; read does not raise; value lands in memory.
        store.record("s1", "eng", 1.5)
        # Broken redis: lrange also raises, so read falls through to memory.
        assert store.get_recent("s1", "eng", 10) == [1.5]
