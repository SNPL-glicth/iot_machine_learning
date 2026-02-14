"""Tests for Phase 3 improvements: E-5, E-7, E-8, E-9, E-11.

E-5:  MetricsCollector wiring (stream consumer + batch runner)
E-7:  Pydantic V2 migration (MQTTReadingPayload)
E-8:  Timestamp re-parse elimination
E-9:  Health/ready endpoints
E-11: Broker latency metrics (BrokerMetrics)
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pathlib

import pytest

_INGEST_ROOT = pathlib.Path(
    "/home/nicolas/Documentos/Iot_System/iot_ingest_services"
)


def _can_import_ingest() -> bool:
    try:
        import iot_ingest_services.jobs.batch.runner  # noqa: F401
        return True
    except Exception:
        return False


_skip_ingest = pytest.mark.skipif(
    not _can_import_ingest(),
    reason="iot_ingest_services not importable in this test context",
)


# ── E-5: MetricsCollector wiring ──────────────────────────────────────


class TestMetricsCollectorWiring:
    """Verify stream consumer and batch runner wire to MetricsCollector."""

    def test_stream_consumer_imports_metrics(self):
        """stream_consumer.py imports MetricsCollector."""
        from iot_machine_learning.ml_service.consumers import stream_consumer

        assert hasattr(stream_consumer, "MetricsCollector")

    def test_batch_runner_imports_metrics(self):
        """runner.py source imports MetricsCollector."""
        src = (_INGEST_ROOT / "jobs/batch/runner.py").read_text()
        assert "from iot_machine_learning.ml_service.metrics.performance_metrics import MetricsCollector" in src

    def test_metrics_collector_singleton(self):
        """MetricsCollector.get_instance() returns same object."""
        from iot_machine_learning.ml_service.metrics.performance_metrics import (
            MetricsCollector,
        )

        a = MetricsCollector.get_instance()
        b = MetricsCollector.get_instance()
        assert a is b

    def test_metrics_collector_records_prediction(self):
        """record_prediction increments counter."""
        from iot_machine_learning.ml_service.metrics.performance_metrics import (
            MetricsCollector,
        )

        mc = MetricsCollector.get_instance()
        before = mc._prediction_count
        mc.record_prediction(1.5)
        assert mc._prediction_count == before + 1

    def test_metrics_collector_records_error(self):
        """record_error increments counter."""
        from iot_machine_learning.ml_service.metrics.performance_metrics import (
            MetricsCollector,
        )

        mc = MetricsCollector.get_instance()
        before = mc._error_count
        mc.record_error()
        assert mc._error_count == before + 1

    def test_metrics_to_dict_structure(self):
        """get_metrics().to_dict() has expected keys."""
        from iot_machine_learning.ml_service.metrics.performance_metrics import (
            MetricsCollector,
        )

        d = MetricsCollector.get_instance().get_metrics().to_dict()
        assert "predictions" in d
        assert "readings" in d
        assert "errors" in d
        assert "persistence" in d
        assert "anomalies" in d
        assert "uptime" in d


# ── E-7: Pydantic V2 migration ───────────────────────────────────────


@_skip_ingest
class TestPydanticV2Migration:
    """Verify MQTTReadingPayload uses Pydantic V2 syntax."""

    def _make_payload(self, **overrides):
        from iot_ingest_services.ingest_api.mqtt.validators import (
            MQTTReadingPayload,
        )

        ts = datetime.now(timezone.utc).isoformat()
        defaults = {
            "sensorId": "42",
            "value": 23.5,
            "timestamp": ts,
        }
        defaults.update(overrides)
        return MQTTReadingPayload(**defaults)

    def test_model_config_exists(self):
        """Uses ConfigDict instead of class Config."""
        from iot_ingest_services.ingest_api.mqtt.validators import (
            MQTTReadingPayload,
        )

        assert hasattr(MQTTReadingPayload, "model_config")

    def test_field_validator_value_nan(self):
        """field_validator rejects NaN."""
        with pytest.raises(Exception):
            self._make_payload(value=float("nan"))

    def test_field_validator_value_inf(self):
        """field_validator rejects infinity."""
        with pytest.raises(Exception):
            self._make_payload(value=float("inf"))

    def test_field_validator_sensor_id_empty(self):
        """field_validator rejects empty sensorId."""
        with pytest.raises(Exception):
            self._make_payload(sensorId="   ")

    def test_model_validator_caches_timestamp(self):
        """model_validator(mode='after') caches parsed timestamp."""
        p = self._make_payload()
        assert p._parsed_ts is not None
        assert p._parsed_dt is not None
        assert isinstance(p._parsed_dt, datetime)

    def test_timestamp_float_uses_cache(self):
        """timestamp_float property returns cached value."""
        p = self._make_payload()
        assert abs(p.timestamp_float - p._parsed_ts) < 0.001

    def test_timestamp_datetime_uses_cache(self):
        """timestamp_datetime property returns cached value."""
        p = self._make_payload()
        assert p.timestamp_datetime is p._parsed_dt

    def test_model_dump_works(self):
        """Pydantic V2 model_dump() works."""
        p = self._make_payload()
        d = p.model_dump()
        assert "sensor_id" in d
        assert "value" in d

    def test_populate_by_name(self):
        """populate_by_name allows both alias and field name."""
        from iot_ingest_services.ingest_api.mqtt.validators import (
            MQTTReadingPayload,
        )

        ts = datetime.now(timezone.utc).isoformat()
        p = MQTTReadingPayload(sensorId="1", value=10.0, timestamp=ts)
        assert p.sensor_id == "1"

    def test_private_attr_not_in_dict(self):
        """PrivateAttr fields are not in model_dump()."""
        p = self._make_payload()
        d = p.model_dump()
        assert "_parsed_ts" not in d
        assert "_parsed_dt" not in d


# ── E-8: Timestamp re-parse elimination ───────────────────────────────


class TestTimestampReparse:
    """Verify processor uses payload.timestamp_datetime directly."""

    def test_processor_no_parse_timestamp_method(self):
        """ReadingProcessor source should not have _parse_timestamp anymore."""
        src = (_INGEST_ROOT / "ingest_api/mqtt/processor.py").read_text()
        assert "def _parse_timestamp" not in src

    def test_processor_uses_payload_timestamp_datetime(self):
        """process() uses payload.timestamp_datetime (cached)."""
        src = (_INGEST_ROOT / "ingest_api/mqtt/processor.py").read_text()
        assert "payload.timestamp_datetime" in src
        assert "_parse_timestamp" not in src


# ── E-9: Health endpoints ─────────────────────────────────────────────


class TestHealthEndpoints:
    """Verify /health, /ready, /metrics endpoints exist."""

    def test_ingest_health_endpoint_exists(self):
        """Ingest health.py has /health route."""
        src = (_INGEST_ROOT / "ingest_api/endpoints/health.py").read_text()
        assert '@router.get("/health")' in src

    def test_ingest_ready_endpoint_exists(self):
        """Ingest health.py has /ready route."""
        src = (_INGEST_ROOT / "ingest_api/endpoints/health.py").read_text()
        assert '@router.get("/ready")' in src

    def test_ingest_metrics_endpoint_exists(self):
        """Ingest health.py has /metrics route."""
        src = (_INGEST_ROOT / "ingest_api/endpoints/health.py").read_text()
        assert '@router.get("/metrics")' in src

    def test_ml_health_endpoint_exists(self):
        """ML service has /health endpoint."""
        from iot_machine_learning.ml_service.api.routes import router

        paths = [r.path for r in router.routes]
        assert "/health" in paths

    def test_ml_ready_endpoint_exists(self):
        """ML service has /ready endpoint."""
        from iot_machine_learning.ml_service.api.routes import router

        paths = [r.path for r in router.routes]
        assert "/ready" in paths

    def test_ml_metrics_endpoint_exists(self):
        """ML service has /ml/metrics endpoint."""
        from iot_machine_learning.ml_service.api.routes import router

        paths = [r.path for r in router.routes]
        assert "/ml/metrics" in paths


@dataclass
class _BrokerMetrics:
    """Mirror of iot_broker.runner.BrokerMetrics for isolated testing."""

    messages_processed: int = 0
    messages_failed: int = 0
    batches_consumed: int = 0
    _latencies_ms: deque = field(default_factory=lambda: deque(maxlen=1000))
    _start_time: float = field(default_factory=time.monotonic)

    def record(self, latency_ms: float) -> None:
        self.messages_processed += 1
        self._latencies_ms.append(latency_ms)

    def record_error(self) -> None:
        self.messages_failed += 1

    def to_dict(self) -> dict:
        lats = list(self._latencies_ms)
        avg = sum(lats) / len(lats) if lats else 0.0
        p99 = sorted(lats)[int(len(lats) * 0.99)] if lats else 0.0
        mx = max(lats) if lats else 0.0
        uptime = time.monotonic() - self._start_time
        return {
            "messages_processed": self.messages_processed,
            "messages_failed": self.messages_failed,
            "batches_consumed": self.batches_consumed,
            "latency_avg_ms": round(avg, 2),
            "latency_p99_ms": round(p99, 2),
            "latency_max_ms": round(mx, 2),
            "uptime_s": round(uptime, 1),
        }


class TestBrokerMetrics:
    """Verify BrokerMetrics tracks latency correctly."""

    def test_initial_state(self):
        m = _BrokerMetrics()
        assert m.messages_processed == 0
        assert m.messages_failed == 0
        assert m.batches_consumed == 0

    def test_record_increments_count(self):
        m = _BrokerMetrics()
        m.record(5.0)
        m.record(10.0)
        assert m.messages_processed == 2

    def test_record_error_increments(self):
        m = _BrokerMetrics()
        m.record_error()
        m.record_error()
        assert m.messages_failed == 2

    def test_to_dict_structure(self):
        m = _BrokerMetrics()
        m.record(2.0)
        m.record(8.0)
        d = m.to_dict()
        assert "messages_processed" in d
        assert "messages_failed" in d
        assert "batches_consumed" in d
        assert "latency_avg_ms" in d
        assert "latency_p99_ms" in d
        assert "latency_max_ms" in d
        assert "uptime_s" in d

    def test_latency_avg(self):
        m = _BrokerMetrics()
        m.record(2.0)
        m.record(4.0)
        m.record(6.0)
        d = m.to_dict()
        assert d["latency_avg_ms"] == 4.0

    def test_latency_max(self):
        m = _BrokerMetrics()
        m.record(1.0)
        m.record(99.0)
        m.record(3.0)
        d = m.to_dict()
        assert d["latency_max_ms"] == 99.0

    def test_latency_p99_single(self):
        m = _BrokerMetrics()
        m.record(5.0)
        d = m.to_dict()
        assert d["latency_p99_ms"] == 5.0

    def test_empty_to_dict(self):
        m = _BrokerMetrics()
        d = m.to_dict()
        assert d["latency_avg_ms"] == 0.0
        assert d["latency_max_ms"] == 0.0
        assert d["latency_p99_ms"] == 0.0

    def test_broker_runner_source_has_metrics(self):
        """BrokerRunner source code references BrokerMetrics."""
        import pathlib
        runner_path = pathlib.Path(
            "/home/nicolas/Documentos/Iot_System/iot_broker/src/iot_broker/runner.py"
        )
        src = runner_path.read_text()
        assert "BrokerMetrics" in src
        assert "self.metrics = BrokerMetrics()" in src
        assert "self.metrics.record(latency_ms)" in src
        assert "self.metrics.record_error()" in src
