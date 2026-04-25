"""Tests for SeriesValuesStore (IMP-1 supporting infra).

Covers:
- Inert mode (no Redis) — all operations are safe no-ops.
- Roundtrip append / get_recent with TTL refresh and list trimming.
- append_many batches and respects max_values.
- NaN/Inf/non-numeric are dropped silently.
- get_bounds returns (lower, upper) when enough samples, None otherwise.
- get_bounds returns None on zero variance.
- Bytes values from real redis-py are decoded on read.
- Redis failures are swallowed.
- Constructor validation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.series_values import (
    SeriesValuesStore,
)


# -- minimal fake Redis ---------------------------------------------------


class _FakePipeline:
    def __init__(self, store: Dict[str, List[str]], ttls: Dict[str, int]) -> None:
        self._store = store
        self._ttls = ttls
        self._ops: List[tuple] = []

    def rpush(self, key: str, *values: str) -> "_FakePipeline":
        self._ops.append(("rpush", key, list(values)))
        return self

    def ltrim(self, key: str, start: int, end: int) -> "_FakePipeline":
        self._ops.append(("ltrim", key, start, end))
        return self

    def expire(self, key: str, ttl: int) -> "_FakePipeline":
        self._ops.append(("expire", key, int(ttl)))
        return self

    def execute(self) -> None:
        for op in self._ops:
            if op[0] == "rpush":
                self._store.setdefault(op[1], []).extend(op[2])
            elif op[0] == "ltrim":
                _, key, start, end = op
                lst = self._store.get(key, [])
                # Emulate Redis ltrim semantics (inclusive, supports negatives)
                if end == -1:
                    sliced = lst[start:] if start < 0 else lst[start:]
                else:
                    sliced = lst[start : end + 1]
                self._store[key] = sliced
            elif op[0] == "expire":
                self._ttls[op[1]] = op[2]


class _FakeRedis:
    def __init__(self) -> None:
        self.store: Dict[str, List[str]] = {}
        self.ttls: Dict[str, int] = {}

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self.store, self.ttls)

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        lst = self.store.get(key, [])
        if end == -1:
            return list(lst[start:])
        return list(lst[start : end + 1])

    def delete(self, key: str) -> int:
        return 1 if self.store.pop(key, None) is not None else 0


class _BrokenRedis:
    def pipeline(self):
        raise RuntimeError("redis down")

    def lrange(self, *_a, **_k):
        raise RuntimeError("redis down")

    def delete(self, *_a, **_k):
        raise RuntimeError("redis down")


# -- inert mode ------------------------------------------------------------


def test_inert_when_no_redis() -> None:
    s = SeriesValuesStore(redis_client=None)
    assert s.is_active is False
    s.append("sid", 1.0)
    s.append_many("sid", [1.0, 2.0])
    assert s.get_recent("sid") == []
    assert s.get_bounds("sid") is None
    s.reset("sid")


# -- roundtrip -------------------------------------------------------------


def test_append_and_get_recent() -> None:
    r = _FakeRedis()
    s = SeriesValuesStore(redis_client=r, ttl_seconds=60)
    for v in [1.0, 2.0, 3.0, 4.0]:
        s.append("sid", v)
    assert s.get_recent("sid") == [1.0, 2.0, 3.0, 4.0]
    assert r.ttls["series_values:sid"] == 60


def test_append_many_and_trim() -> None:
    r = _FakeRedis()
    s = SeriesValuesStore(redis_client=r, max_values=3)
    s.append_many("sid", [1.0, 2.0, 3.0, 4.0, 5.0])
    # Only last 3 retained
    assert s.get_recent("sid") == [3.0, 4.0, 5.0]


def test_get_recent_limit() -> None:
    r = _FakeRedis()
    s = SeriesValuesStore(redis_client=r)
    s.append_many("sid", [1.0, 2.0, 3.0, 4.0, 5.0])
    assert s.get_recent("sid", n=3) == [3.0, 4.0, 5.0]
    assert s.get_recent("sid", n=0) == [1.0, 2.0, 3.0, 4.0, 5.0]  # 0 ignored


def test_reset_deletes_key() -> None:
    r = _FakeRedis()
    s = SeriesValuesStore(redis_client=r)
    s.append("sid", 1.0)
    s.reset("sid")
    assert s.get_recent("sid") == []


# -- filtering -------------------------------------------------------------


def test_nonfinite_values_dropped_on_write() -> None:
    r = _FakeRedis()
    s = SeriesValuesStore(redis_client=r)
    s.append("sid", float("nan"))
    s.append("sid", float("inf"))
    s.append("sid", float("-inf"))
    s.append("sid", "garbage")  # type: ignore[arg-type]
    assert s.get_recent("sid") == []


def test_append_many_filters_nonfinite() -> None:
    r = _FakeRedis()
    s = SeriesValuesStore(redis_client=r)
    s.append_many("sid", [1.0, float("nan"), 2.0, float("inf"), 3.0])
    assert s.get_recent("sid") == [1.0, 2.0, 3.0]


def test_bytes_decoded_on_read() -> None:
    r = _FakeRedis()
    r.store["series_values:sid"] = [b"1.0", b"2.5", b"bad", b"3.0"]
    s = SeriesValuesStore(redis_client=r)
    assert s.get_recent("sid") == [1.0, 2.5, 3.0]


# -- bounds ----------------------------------------------------------------


def test_get_bounds_insufficient_samples() -> None:
    r = _FakeRedis()
    s = SeriesValuesStore(redis_client=r)
    s.append_many("sid", [1.0, 2.0, 3.0])  # only 3 samples
    assert s.get_bounds("sid", min_samples=20) is None


def test_get_bounds_happy_path() -> None:
    r = _FakeRedis()
    s = SeriesValuesStore(redis_client=r)
    values = [10.0 + (i % 5) * 0.1 for i in range(30)]  # mean~10.2
    s.append_many("sid", values)
    bounds = s.get_bounds("sid", sigma_multiplier=6.0, min_samples=20)
    assert bounds is not None
    lower, upper = bounds
    mean = sum(values) / len(values)
    assert lower < mean < upper
    assert (upper - lower) > 0


def test_get_bounds_zero_variance_returns_none() -> None:
    r = _FakeRedis()
    s = SeriesValuesStore(redis_client=r)
    s.append_many("sid", [7.0] * 30)
    assert s.get_bounds("sid", min_samples=20) is None


# -- defensive -------------------------------------------------------------


def test_redis_failure_is_swallowed() -> None:
    s = SeriesValuesStore(redis_client=_BrokenRedis())
    s.append("sid", 1.0)
    s.append_many("sid", [1.0, 2.0])
    assert s.get_recent("sid") == []
    assert s.get_bounds("sid") is None
    s.reset("sid")


def test_blank_series_id_is_noop() -> None:
    r = _FakeRedis()
    s = SeriesValuesStore(redis_client=r)
    s.append("", 1.0)
    s.append_many("", [1.0])
    assert r.store == {}
    assert s.get_recent("") == []
    assert s.get_bounds("") is None


# -- constructor ----------------------------------------------------------


def test_invalid_ttl_raises() -> None:
    with pytest.raises(ValueError):
        SeriesValuesStore(ttl_seconds=0)


def test_invalid_max_values_raises() -> None:
    with pytest.raises(ValueError):
        SeriesValuesStore(max_values=0)


def test_invalid_prefix_raises() -> None:
    with pytest.raises(ValueError):
        SeriesValuesStore(key_prefix="")
