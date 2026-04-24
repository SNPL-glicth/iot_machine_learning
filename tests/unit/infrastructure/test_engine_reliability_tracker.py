"""Tests for EngineReliabilityTracker (IMP-4b).

Covers:
1. Prior state is reliable (uninformative Beta(1,1) → P(broken)=0.5 < 0.95).
2. Clean outcomes keep engine reliable; alpha grows.
3. A burst of high errors pushes P(broken) above 0.95 → is_reliable=False.
4. Recovery: after an unreliable burst, clean outcomes bring is_reliable back to True.
5. No history in EngineErrorStore → record_outcome is a silent no-op.
6. Bad inputs (NaN, negative, non-numeric) → silent no-op.
7. reset() restores the prior.
8. Redis backend happy path via fake pipeline.
9. Redis read failure → falls back to in-memory prior (no exception).
10. Invalid constructor arguments raise ValueError.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.error_store import (
    EngineErrorStore,
)
from iot_machine_learning.infrastructure.ml.cognitive.reliability import (
    EngineReliabilityTracker,
)


# -- Fake Redis ------------------------------------------------------------


class _FakePipeline:
    def __init__(self, store: Dict[str, Dict[str, str]]) -> None:
        self._store = store
        self._ops: List[tuple] = []

    def hset(self, key: str, mapping: Dict[str, Any]) -> None:
        self._ops.append(("hset", key, mapping))

    def expire(self, key: str, ttl: int) -> None:
        self._ops.append(("expire", key, ttl))

    def rpush(self, key, value):  # noqa: D401 \u2014 for error-store fake
        self._ops.append(("rpush", key, value))

    def ltrim(self, key, start, end):
        self._ops.append(("ltrim", key, start, end))

    def execute(self) -> None:
        for op in self._ops:
            if op[0] == "hset":
                self._store[op[1]] = {k: str(v) for k, v in op[2].items()}
            elif op[0] == "rpush":
                self._store.setdefault(op[1], []).append(str(op[2]))
            elif op[0] == "ltrim":
                pass


class _FakeRedis:
    def __init__(self) -> None:
        self.store: Dict[str, Any] = {}

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self.store)

    def hgetall(self, key: str) -> Dict[str, str]:
        val = self.store.get(key, {})
        return val if isinstance(val, dict) else {}

    def lrange(self, key: str, start: int, end: int) -> List[str]:
        lst = self.store.get(key, [])
        if not isinstance(lst, list):
            return []
        return lst if end == -1 else lst[start:end + 1]


class _BrokenRedis:
    def hgetall(self, *_a, **_k):  # pragma: no cover
        raise RuntimeError("redis down")

    def pipeline(self):  # pragma: no cover
        raise RuntimeError("redis down")


# -- Fixtures --------------------------------------------------------------


def _store_with_history(errors: List[float]) -> EngineErrorStore:
    store = EngineErrorStore()
    for e in errors:
        store.record("s1", "eng", e)
    return store


# -- Core semantics --------------------------------------------------------


class TestPriorAndUpdates:
    def test_prior_is_reliable_by_default(self) -> None:
        store = _store_with_history([1.0] * 20)
        tracker = EngineReliabilityTracker(store)
        assert tracker.is_reliable("s1", "eng") is True
        assert tracker.p_broken("s1", "eng") == pytest.approx(0.5)

    def test_clean_outcomes_keep_reliable(self) -> None:
        store = _store_with_history([1.0] * 20)  # p75 \u2248 1.0
        tracker = EngineReliabilityTracker(store)
        for _ in range(20):
            tracker.record_outcome("s1", "eng", 0.5)  # below threshold
        assert tracker.is_reliable("s1", "eng") is True
        assert tracker.p_broken("s1", "eng") < 0.1

    def test_bad_burst_flips_to_unreliable(self) -> None:
        store = _store_with_history([1.0] * 20)
        tracker = EngineReliabilityTracker(store)
        # 200 consecutive "broken" outcomes \u2014 P(broken) \u2248 201/202 > 0.95
        for _ in range(200):
            tracker.record_outcome("s1", "eng", 99.0)  # above threshold
        assert tracker.is_reliable("s1", "eng") is False
        assert tracker.p_broken("s1", "eng") > 0.95

    def test_recovery_after_unreliable(self) -> None:
        store = _store_with_history([1.0] * 20)
        tracker = EngineReliabilityTracker(store)
        for _ in range(40):
            tracker.record_outcome("s1", "eng", 99.0)
        assert tracker.is_reliable("s1", "eng") is False
        # Now many clean outcomes. Each adds to \u03b1 (error 0.1 < threshold 1.0).
        for _ in range(2000):
            tracker.record_outcome("s1", "eng", 0.1)
        assert tracker.is_reliable("s1", "eng") is True


# -- Defensive paths -------------------------------------------------------


class TestDefensive:
    def test_empty_history_is_noop(self) -> None:
        store = EngineErrorStore()  # no history
        tracker = EngineReliabilityTracker(store)
        tracker.record_outcome("s1", "eng", 99.0)  # should not change state
        assert tracker.p_broken("s1", "eng") == pytest.approx(0.5)

    def test_invalid_errors_are_ignored(self) -> None:
        store = _store_with_history([1.0] * 20)
        tracker = EngineReliabilityTracker(store)
        tracker.record_outcome("s1", "eng", float("nan"))
        tracker.record_outcome("s1", "eng", -1.0)
        tracker.record_outcome("s1", "eng", "bad")  # type: ignore[arg-type]
        # prior intact
        assert tracker.p_broken("s1", "eng") == pytest.approx(0.5)

    def test_reset_restores_prior(self) -> None:
        store = _store_with_history([1.0] * 20)
        tracker = EngineReliabilityTracker(store)
        for _ in range(50):
            tracker.record_outcome("s1", "eng", 99.0)
        assert tracker.p_broken("s1", "eng") > 0.9
        tracker.reset("s1", "eng")
        assert tracker.p_broken("s1", "eng") == pytest.approx(0.5)


# -- Redis backend ---------------------------------------------------------


class TestRedisBackend:
    def test_redis_roundtrip(self) -> None:
        fake = _FakeRedis()
        store = EngineErrorStore(redis_client=fake, key_prefix="es", ttl_seconds=60)
        for e in [1.0] * 20:
            store.record("s1", "eng", e)
        tracker = EngineReliabilityTracker(
            store, redis_client=fake, key_prefix="rel", ttl_seconds=60
        )
        for _ in range(20):
            tracker.record_outcome("s1", "eng", 0.1)
        assert "rel:s1:eng" in fake.store
        assert float(fake.store["rel:s1:eng"]["alpha"]) > 1.0

    def test_redis_read_failure_falls_back(self) -> None:
        store = _store_with_history([1.0] * 20)
        tracker = EngineReliabilityTracker(store, redis_client=_BrokenRedis())
        # Broken redis on both read and write \u2014 tracker stays on in-memory path.
        assert tracker.p_broken("s1", "eng") == pytest.approx(0.5)


# -- Validation ------------------------------------------------------------


class TestConstructorValidation:
    def test_invalid_percentile(self) -> None:
        store = EngineErrorStore()
        with pytest.raises(ValueError):
            EngineReliabilityTracker(store, percentile=0.0)
        with pytest.raises(ValueError):
            EngineReliabilityTracker(store, percentile=100.0)

    def test_invalid_threshold(self) -> None:
        store = EngineErrorStore()
        with pytest.raises(ValueError):
            EngineReliabilityTracker(store, unreliable_threshold=0.0)
        with pytest.raises(ValueError):
            EngineReliabilityTracker(store, unreliable_threshold=1.0)

    def test_invalid_priors(self) -> None:
        store = EngineErrorStore()
        with pytest.raises(ValueError):
            EngineReliabilityTracker(store, alpha_prior=0.0)
        with pytest.raises(ValueError):
            EngineReliabilityTracker(store, beta_prior=-1.0)
