"""Tests for HyperparameterAdaptor (IMP-4c).

Covers:
1. Inert mode when redis_client is None (load=None, save/reset no-op, is_active=False).
2. Redis round-trip happy path with a fake pipeline.
3. TTL refresh on every save.
4. reset() deletes the key.
5. Malformed / non-finite / non-numeric values are filtered on encode.
6. Bytes keys/values coming from real redis are decoded on load.
7. Redis read/write failures are swallowed and never raise.
8. Constructor rejects invalid ttl or prefix.
"""

from __future__ import annotations

from typing import Any, Dict, List

import math
import pytest

from iot_machine_learning.infrastructure.ml.cognitive.hyperparameters import (
    HyperparameterAdaptor,
)


class _FakePipeline:
    def __init__(self, store: Dict[str, Dict[str, str]], ttls: Dict[str, int]) -> None:
        self._store = store
        self._ttls = ttls
        self._ops: List[tuple] = []

    def hset(self, key: str, mapping: Dict[str, Any]) -> "_FakePipeline":
        self._ops.append(("hset", key, dict(mapping)))
        return self

    def expire(self, key: str, ttl: int) -> "_FakePipeline":
        self._ops.append(("expire", key, int(ttl)))
        return self

    def execute(self) -> None:
        for op in self._ops:
            if op[0] == "hset":
                self._store[op[1]] = {k: str(v) for k, v in op[2].items()}
            elif op[0] == "expire":
                self._ttls[op[1]] = op[2]


class _FakeRedis:
    def __init__(self) -> None:
        self.store: Dict[str, Dict[str, str]] = {}
        self.ttls: Dict[str, int] = {}

    def pipeline(self) -> _FakePipeline:
        return _FakePipeline(self.store, self.ttls)

    def hgetall(self, key: str) -> Dict[str, str]:
        return dict(self.store.get(key, {}))

    def delete(self, key: str) -> int:
        existed = key in self.store
        self.store.pop(key, None)
        self.ttls.pop(key, None)
        return 1 if existed else 0


class _BrokenRedis:
    def pipeline(self):
        raise RuntimeError("redis down")

    def hgetall(self, *_a, **_k):
        raise RuntimeError("redis down")

    def delete(self, *_a, **_k):
        raise RuntimeError("redis down")


# -- inert mode ------------------------------------------------------------


def test_inert_when_no_redis() -> None:
    a = HyperparameterAdaptor(redis_client=None)
    assert a.is_active is False
    assert a.load("s", "e") is None
    a.save("s", "e", {"alpha": 0.5})  # no raise
    a.reset("s", "e")  # no raise
    # Still inert after operations
    assert a.load("s", "e") is None


# -- Redis round-trip ------------------------------------------------------


def test_save_then_load_roundtrip() -> None:
    r = _FakeRedis()
    a = HyperparameterAdaptor(redis_client=r, ttl_seconds=60)
    a.save("series-1", "statistical_ema_holt", {"alpha": 0.42, "beta": 0.13})
    loaded = a.load("series-1", "statistical_ema_holt")
    assert loaded == {"alpha": 0.42, "beta": 0.13}
    assert a.is_active is True


def test_load_missing_returns_none() -> None:
    a = HyperparameterAdaptor(redis_client=_FakeRedis())
    assert a.load("unknown", "engine") is None


def test_ttl_is_refreshed_on_every_save() -> None:
    r = _FakeRedis()
    a = HyperparameterAdaptor(redis_client=r, ttl_seconds=30)
    a.save("s", "e", {"x": 1.0})
    a.save("s", "e", {"x": 2.0})
    assert r.ttls["engine_hyperparams:s:e"] == 30


def test_reset_deletes_key() -> None:
    r = _FakeRedis()
    a = HyperparameterAdaptor(redis_client=r)
    a.save("s", "e", {"x": 1.0})
    a.reset("s", "e")
    assert a.load("s", "e") is None


# -- encoding / decoding ---------------------------------------------------


def test_non_finite_and_nonnumeric_are_filtered() -> None:
    r = _FakeRedis()
    a = HyperparameterAdaptor(redis_client=r)
    a.save("s", "e", {
        "ok": 0.5,
        "nan": float("nan"),
        "inf": float("inf"),
        "str": "oops",  # type: ignore[arg-type]
        "": 1.0,
    })
    loaded = a.load("s", "e")
    assert loaded == {"ok": 0.5}


def test_empty_params_skipped() -> None:
    r = _FakeRedis()
    a = HyperparameterAdaptor(redis_client=r)
    a.save("s", "e", {})
    # No key written → load returns None
    assert a.load("s", "e") is None


def test_bytes_keys_and_values_decoded_on_load() -> None:
    r = _FakeRedis()
    # Simulate a real redis-py response (bytes everywhere)
    r.store["engine_hyperparams:s:e"] = {b"alpha": b"0.7", b"beta": b"0.2"}
    a = HyperparameterAdaptor(redis_client=r)
    loaded = a.load("s", "e")
    assert loaded == {"alpha": 0.7, "beta": 0.2}


def test_malformed_stored_values_filtered_on_load() -> None:
    r = _FakeRedis()
    r.store["engine_hyperparams:s:e"] = {"alpha": "0.5", "beta": "nope", "gamma": "nan"}
    a = HyperparameterAdaptor(redis_client=r)
    loaded = a.load("s", "e")
    assert loaded == {"alpha": 0.5}


# -- defensive paths -------------------------------------------------------


def test_redis_failure_is_swallowed() -> None:
    a = HyperparameterAdaptor(redis_client=_BrokenRedis())
    # None of these should raise
    assert a.load("s", "e") is None
    a.save("s", "e", {"x": 1.0})
    a.reset("s", "e")


def test_blank_identifiers_are_noops() -> None:
    r = _FakeRedis()
    a = HyperparameterAdaptor(redis_client=r)
    a.save("", "e", {"x": 1.0})
    a.save("s", "", {"x": 1.0})
    assert a.load("", "e") is None
    assert a.load("s", "") is None
    assert r.store == {}


# -- constructor validation ------------------------------------------------


def test_invalid_ttl_raises() -> None:
    with pytest.raises(ValueError):
        HyperparameterAdaptor(ttl_seconds=0)


def test_invalid_prefix_raises() -> None:
    with pytest.raises(ValueError):
        HyperparameterAdaptor(key_prefix="")
