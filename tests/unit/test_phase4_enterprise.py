"""Tests for Phase 4 improvements: E-6, E-13, E-15.

E-6:  Canonical reading type + adapters
E-13: Service separation facades (ml_api, ml_batch, ml_stream)
E-15: Consolidated sliding window (ISlidingWindowStore + InMemorySlidingWindowStore)
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pytest

from iot_machine_learning.domain.entities.iot.sensor_reading import (
    SensorReading,
    SensorWindow,
)


# ── E-15: Consolidated Sliding Window ─────────────────────────────────


class TestISlidingWindowStorePort:
    """Verify the ISlidingWindowStore interface exists and is well-defined."""

    def test_port_importable(self):
        from iot_machine_learning.domain.ports.sliding_window_port import (
            ISlidingWindowStore,
            WindowConfig,
        )
        assert ISlidingWindowStore is not None
        assert WindowConfig is not None

    def test_port_in_ports_package(self):
        from iot_machine_learning.domain.ports import (
            ISlidingWindowStore,
            WindowConfig,
        )
        assert ISlidingWindowStore is not None
        assert WindowConfig is not None

    def test_window_config_defaults(self):
        from iot_machine_learning.domain.ports.sliding_window_port import WindowConfig
        cfg = WindowConfig()
        assert cfg.max_size == 100
        assert cfg.max_age_seconds == 3600.0
        assert cfg.max_sensors == 1000
        assert cfg.ttl_seconds == 3600.0

    def test_window_config_custom(self):
        from iot_machine_learning.domain.ports.sliding_window_port import WindowConfig
        cfg = WindowConfig(max_size=50, max_age_seconds=600, max_sensors=200, ttl_seconds=1800)
        assert cfg.max_size == 50
        assert cfg.max_sensors == 200

    def test_port_is_abstract(self):
        from iot_machine_learning.domain.ports.sliding_window_port import ISlidingWindowStore
        with pytest.raises(TypeError):
            ISlidingWindowStore()  # type: ignore[abstract]


class TestInMemorySlidingWindowStore:
    """Verify the canonical InMemorySlidingWindowStore implementation."""

    def _make_store(self, **kwargs):
        from iot_machine_learning.domain.ports.sliding_window_port import WindowConfig
        from iot_machine_learning.infrastructure.sliding_window.in_memory import (
            InMemorySlidingWindowStore,
        )
        cfg = WindowConfig(**kwargs)
        return InMemorySlidingWindowStore(cfg)

    def test_implements_interface(self):
        from iot_machine_learning.domain.ports.sliding_window_port import ISlidingWindowStore
        store = self._make_store()
        assert isinstance(store, ISlidingWindowStore)

    def test_append_and_get(self):
        store = self._make_store()
        store.append(1, "a", 1.0)
        store.append(1, "b", 2.0)
        assert store.get_window(1) == ["a", "b"]

    def test_append_returns_size(self):
        store = self._make_store()
        assert store.append(1, "x", 1.0) == 1
        assert store.append(1, "y", 2.0) == 2

    def test_get_window_empty(self):
        store = self._make_store()
        assert store.get_window(999) == []

    def test_get_size(self):
        store = self._make_store()
        store.append(1, "a", 1.0)
        store.append(1, "b", 2.0)
        assert store.get_size(1) == 2
        assert store.get_size(999) == 0

    def test_clear(self):
        store = self._make_store()
        store.append(1, "a", 1.0)
        store.clear(1)
        assert store.get_size(1) == 0

    def test_sensor_ids(self):
        store = self._make_store()
        store.append(10, "a", 1.0)
        store.append(20, "b", 2.0)
        ids = store.sensor_ids()
        assert set(ids) == {10, 20}

    def test_max_size_eviction(self):
        store = self._make_store(max_size=3)
        for i in range(5):
            store.append(1, f"item{i}", float(i))
        assert store.get_size(1) == 3
        assert store.get_window(1) == ["item2", "item3", "item4"]

    def test_lru_eviction(self):
        store = self._make_store(max_sensors=2)
        store.append(1, "a", 1.0)
        store.append(2, "b", 2.0)
        store.append(3, "c", 3.0)  # evicts sensor 1
        assert store.get_size(1) == 0
        assert store.get_size(2) > 0
        assert store.get_size(3) > 0

    def test_ttl_eviction(self):
        store = self._make_store(ttl_seconds=0.01)
        store.append(1, "a", 1.0)
        time.sleep(0.02)
        evicted = store.evict_stale(time.monotonic())
        assert evicted >= 1
        assert store.get_size(1) == 0

    def test_get_metrics(self):
        store = self._make_store(max_sensors=2)
        store.append(1, "a", 1.0)
        store.append(2, "b", 2.0)
        store.append(3, "c", 3.0)  # triggers LRU eviction
        m = store.get_metrics()
        assert m["active_sensors"] == 2
        assert m["evictions_lru"] >= 1
        assert "max_sensors" in m
        assert "max_size_per_sensor" in m

    def test_ordered_by_timestamp(self):
        """Items returned in timestamp order even if appended out of order."""
        store = self._make_store()
        store.append(1, "late", 3.0)
        store.append(1, "early", 1.0)
        store.append(1, "mid", 2.0)
        assert store.get_window(1) == ["early", "mid", "late"]

    def test_thread_safety(self):
        """Basic thread safety: concurrent appends don't crash."""
        import threading
        store = self._make_store()
        errors = []

        def worker(sid):
            try:
                for i in range(100):
                    store.append(sid, f"v{i}", float(i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert len(store.sensor_ids()) == 4

    def test_with_sensor_reading(self):
        """Works with SensorReading as the item type."""
        store = self._make_store(max_size=10)
        r1 = SensorReading(sensor_id=1, value=10.0, timestamp=1.0)
        r2 = SensorReading(sensor_id=1, value=20.0, timestamp=2.0)
        store.append(1, r1, r1.timestamp)
        store.append(1, r2, r2.timestamp)
        window = store.get_window(1)
        assert len(window) == 2
        assert window[0].value == 10.0
        assert window[1].value == 20.0


# ── E-6: Canonical Reading Type + Adapters ────────────────────────────


class TestReadingAdapters:
    """Verify adapters convert various reading types to SensorReading."""

    def test_from_stream_reading(self):
        from iot_machine_learning.infrastructure.adapters.reading_adapters import (
            from_stream_reading,
        )

        @dataclass
        class StreamReading:
            sensor_id: int
            value: float
            timestamp: float
            timestamp_iso: str

        r = StreamReading(sensor_id=42, value=23.5, timestamp=1000.0, timestamp_iso="2026-01-01T00:00:00Z")
        sr = from_stream_reading(r)
        assert isinstance(sr, SensorReading)
        assert sr.sensor_id == 42
        assert sr.value == 23.5
        assert sr.timestamp == 1000.0

    def test_from_broker_reading(self):
        from iot_machine_learning.infrastructure.adapters.reading_adapters import (
            from_broker_reading,
        )

        @dataclass(frozen=True)
        class BrokerReading:
            sensor_id: int
            value: float
            timestamp: float
            device_uuid: str = ""
            sensor_uuid: str = ""
            msg_id: str = ""

        r = BrokerReading(sensor_id=10, value=99.9, timestamp=2000.0)
        sr = from_broker_reading(r)
        assert isinstance(sr, SensorReading)
        assert sr.sensor_id == 10
        assert sr.value == 99.9

    def test_from_reading_broker(self):
        from iot_machine_learning.infrastructure.adapters.reading_adapters import (
            from_reading_broker,
        )

        @dataclass(frozen=True)
        class RBReading:
            sensor_id: int
            sensor_type: str
            value: float
            timestamp: float

        r = RBReading(sensor_id=5, sensor_type="temperature", value=22.0, timestamp=3000.0)
        sr = from_reading_broker(r)
        assert isinstance(sr, SensorReading)
        assert sr.sensor_type == "temperature"

    def test_from_dict(self):
        from iot_machine_learning.infrastructure.adapters.reading_adapters import from_dict

        d = {"sensor_id": "7", "value": "15.5", "timestamp": "4000.0"}
        sr = from_dict(d)
        assert isinstance(sr, SensorReading)
        assert sr.sensor_id == 7
        assert sr.value == 15.5

    def test_from_dict_with_optional_fields(self):
        from iot_machine_learning.infrastructure.adapters.reading_adapters import from_dict

        d = {
            "sensor_id": 1,
            "value": 10.0,
            "timestamp": 5000.0,
            "sensor_type": "humidity",
            "device_id": 99,
        }
        sr = from_dict(d)
        assert sr.sensor_type == "humidity"
        assert sr.device_id == 99

    def test_from_mqtt_payload_returns_none_for_non_numeric(self):
        from iot_machine_learning.infrastructure.adapters.reading_adapters import (
            from_mqtt_payload,
        )

        class FakePayload:
            sensor_id = "abc"
            sensor_id_int = None
            value = 10.0
            timestamp_float = 1000.0
            sensor_type = None

        result = from_mqtt_payload(FakePayload())
        assert result is None

    def test_from_mqtt_payload_success(self):
        from iot_machine_learning.infrastructure.adapters.reading_adapters import (
            from_mqtt_payload,
        )

        class FakePayload:
            sensor_id = "42"
            sensor_id_int = 42
            value = 23.5
            timestamp_float = 6000.0
            sensor_type = "temp"

        sr = from_mqtt_payload(FakePayload())
        assert isinstance(sr, SensorReading)
        assert sr.sensor_id == 42
        assert sr.sensor_type == "temp"


# ── E-13: Service Separation Facades ──────────────────────────────────


class TestServiceFacades:
    """Verify ml_api, ml_batch, ml_stream facades are importable."""

    def test_ml_api_importable(self):
        from iot_machine_learning.ml_api import create_app
        assert callable(create_app)

    def test_ml_api_creates_fastapi_app(self):
        from iot_machine_learning.ml_api import create_app
        app = create_app()
        assert hasattr(app, "routes")
        # Verify it has the expected routes
        paths = [r.path for r in app.routes]
        assert "/health" in paths

    def test_ml_batch_importable(self):
        from iot_machine_learning.ml_batch import run_batch_cycle
        assert callable(run_batch_cycle)

    def test_ml_stream_importable(self):
        from iot_machine_learning.ml_stream import start_consumer
        assert callable(start_consumer)

    def test_facades_are_packages(self):
        """Each facade is a proper Python package."""
        import pathlib
        base = pathlib.Path("/home/nicolas/Documentos/Iot_System/iot_machine_learning")
        assert (base / "ml_api" / "__init__.py").exists()
        assert (base / "ml_batch" / "__init__.py").exists()
        assert (base / "ml_stream" / "__init__.py").exists()
