"""Tests for LRU + TTL eviction in SlidingWindowStore and SlidingWindowBuffer.

Covers:
- LRU eviction when max_sensors exceeded
- TTL eviction of inactive sensors
- Thread-safety under concurrent access
- Metrics tracking
- Backward-compatible API
"""

from __future__ import annotations

import threading
import time
from unittest.mock import patch

import pytest

from iot_machine_learning.ml_service.consumers.sliding_window import (
    Reading,
    SlidingWindowStore,
)
from iot_machine_learning.ml_service.sliding_window_buffer import (
    SlidingWindowBuffer,
    WindowStats,
)


# ======================================================================
# Helpers
# ======================================================================

def _reading(sensor_id: int, value: float = 1.0, ts: float = 1000.0) -> Reading:
    return Reading(
        sensor_id=sensor_id, value=value,
        timestamp=ts, timestamp_iso="2026-01-01T00:00:00Z",
    )


# ======================================================================
# SlidingWindowStore — LRU
# ======================================================================

class TestSlidingWindowStoreLRU:

    def test_lru_eviction_oldest_removed(self):
        store = SlidingWindowStore(max_size=20, max_sensors=3)
        store.append(_reading(1, ts=100))
        store.append(_reading(2, ts=101))
        store.append(_reading(3, ts=102))

        # Adding sensor 4 should evict sensor 1 (oldest)
        store.append(_reading(4, ts=103))

        assert store.get_window(1) == []
        assert len(store.get_window(4)) == 1
        assert store.get_metrics()["evictions_lru"] == 1

    def test_lru_access_refreshes_order(self):
        store = SlidingWindowStore(max_size=20, max_sensors=3)
        store.append(_reading(1, ts=100))
        store.append(_reading(2, ts=101))
        store.append(_reading(3, ts=102))

        # Access sensor 1 → moves to end of LRU
        store.get_window(1)

        # Adding sensor 4 should evict sensor 2 (now oldest)
        store.append(_reading(4, ts=103))

        assert len(store.get_window(1)) == 1  # still alive
        assert store.get_window(2) == []       # evicted
        assert store.get_metrics()["evictions_lru"] == 1

    def test_existing_sensor_does_not_trigger_eviction(self):
        store = SlidingWindowStore(max_size=20, max_sensors=2)
        store.append(_reading(1, ts=100))
        store.append(_reading(2, ts=101))

        # Re-adding to existing sensor should NOT evict
        store.append(_reading(1, ts=102))

        assert len(store.get_window(1)) == 2
        assert len(store.get_window(2)) == 1
        assert store.get_metrics()["evictions_lru"] == 0

    def test_max_sensors_one(self):
        store = SlidingWindowStore(max_size=5, max_sensors=1)
        store.append(_reading(1, ts=100))
        store.append(_reading(2, ts=101))

        assert store.get_window(1) == []
        assert len(store.get_window(2)) == 1
        assert store.get_metrics()["evictions_lru"] == 1


# ======================================================================
# SlidingWindowStore — TTL
# ======================================================================

class TestSlidingWindowStoreTTL:

    def test_ttl_eviction_expired_sensor(self):
        store = SlidingWindowStore(max_size=20, max_sensors=1000, ttl_seconds=0.1)
        store.append(_reading(1, ts=100))

        # Wait for TTL to expire
        time.sleep(0.15)

        # Next append triggers cleanup
        store.append(_reading(2, ts=200))

        assert store.get_window(1) == []
        assert len(store.get_window(2)) == 1
        assert store.get_metrics()["evictions_ttl"] == 1

    def test_ttl_active_sensor_not_evicted(self):
        store = SlidingWindowStore(max_size=20, max_sensors=1000, ttl_seconds=0.5)
        store.append(_reading(1, ts=100))

        # Access before TTL
        time.sleep(0.1)
        store.append(_reading(1, ts=101))

        # Trigger cleanup via another sensor
        time.sleep(0.1)
        store.append(_reading(2, ts=200))

        assert len(store.get_window(1)) == 2
        assert store.get_metrics()["evictions_ttl"] == 0


# ======================================================================
# SlidingWindowStore — Thread Safety
# ======================================================================

class TestSlidingWindowStoreThreadSafety:

    def test_concurrent_appends(self):
        store = SlidingWindowStore(max_size=100, max_sensors=50)
        errors = []

        def writer(sensor_id: int):
            try:
                for i in range(100):
                    store.append(_reading(sensor_id, value=float(i), ts=1000.0 + i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(sid,)) for sid in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert store.get_metrics()["active_sensors"] == 10

    def test_concurrent_read_write(self):
        store = SlidingWindowStore(max_size=20, max_sensors=100)
        errors = []

        def writer():
            try:
                for i in range(200):
                    store.append(_reading(i % 5, value=float(i), ts=1000.0 + i))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(200):
                    store.get_window(0)
                    store.sensor_ids
                    store.get_metrics()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert errors == []


# ======================================================================
# SlidingWindowStore — Backward Compatibility
# ======================================================================

class TestSlidingWindowStoreCompat:

    def test_default_constructor(self):
        """Old code: SlidingWindowStore(max_size=20) still works."""
        store = SlidingWindowStore(max_size=20)
        store.append(_reading(1, ts=100))
        window = store.get_window(1)
        assert len(window) == 1
        assert window[0].value == 1.0

    def test_clear(self):
        store = SlidingWindowStore(max_size=20)
        store.append(_reading(1, ts=100))
        store.clear(1)
        assert store.get_window(1) == []

    def test_sensor_ids(self):
        store = SlidingWindowStore(max_size=20)
        store.append(_reading(1, ts=100))
        store.append(_reading(2, ts=101))
        assert set(store.sensor_ids) == {1, 2}

    def test_get_window_nonexistent_returns_empty(self):
        store = SlidingWindowStore(max_size=20)
        assert store.get_window(999) == []

    def test_get_window_sorted_by_timestamp(self):
        store = SlidingWindowStore(max_size=20)
        store.append(_reading(1, value=3.0, ts=103))
        store.append(_reading(1, value=1.0, ts=101))
        store.append(_reading(1, value=2.0, ts=102))
        window = store.get_window(1)
        assert [r.timestamp for r in window] == [101, 102, 103]


# ======================================================================
# SlidingWindowBuffer — LRU + TTL
# ======================================================================

class TestSlidingWindowBufferLRU:

    def test_lru_eviction(self):
        buf = SlidingWindowBuffer(max_horizon_seconds=10.0, max_sensors=3)
        buf.add_reading(1, 10.0, 1000.0)
        buf.add_reading(2, 20.0, 1001.0)
        buf.add_reading(3, 30.0, 1002.0)

        # Sensor 4 should evict sensor 1
        buf.add_reading(4, 40.0, 1003.0)

        metrics = buf.get_metrics()
        assert metrics["evictions_lru"] == 1
        assert metrics["active_sensors"] == 3

    def test_default_constructor_backward_compat(self):
        """Old code: SlidingWindowBuffer(max_horizon_seconds=10.0) still works."""
        buf = SlidingWindowBuffer(max_horizon_seconds=10.0)
        stats = buf.add_reading(1, 25.0, 1000.0)
        assert "w1" in stats or "w5" in stats or "w10" in stats


class TestSlidingWindowBufferTTL:

    def test_ttl_eviction(self):
        buf = SlidingWindowBuffer(
            max_horizon_seconds=10.0, max_sensors=1000, ttl_seconds=0.1,
        )
        buf.add_reading(1, 10.0, 1000.0)

        time.sleep(0.15)

        buf.add_reading(2, 20.0, 2000.0)

        metrics = buf.get_metrics()
        assert metrics["evictions_ttl"] == 1


class TestSlidingWindowBufferMetrics:

    def test_metrics_structure(self):
        buf = SlidingWindowBuffer(max_horizon_seconds=10.0, max_sensors=500)
        m = buf.get_metrics()
        assert m["active_sensors"] == 0
        assert m["max_sensors"] == 500
        assert m["evictions_lru"] == 0
        assert m["evictions_ttl"] == 0
