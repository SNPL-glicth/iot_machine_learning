"""Tests for circuit breaker integration in ReadingProcessor.

Covers:
- Processor works without circuit breaker (backward compat)
- Circuit breaker opens after N failures
- Readings go to DLQ when circuit is open
- Circuit recovers after timeout
- DLQ unavailable → graceful degradation
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

try:
    from iot_ingest_services.ingest_api.pipelines.resilience.circuit_breaker import (
        CircuitBreaker,
    )
    from iot_ingest_services.ingest_api.pipelines.resilience.circuit_breaker_config import (
        CircuitBreakerConfig,
        CircuitBreakerOpen,
        CircuitState,
    )
    from iot_ingest_services.ingest_api.pipelines.resilience.dead_letter import (
        DeadLetterQueue,
    )
    from iot_ingest_services.ingest_api.mqtt.processor import ReadingProcessor
    _INGEST_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _INGEST_AVAILABLE = False


pytestmark = pytest.mark.skipif(
    not _INGEST_AVAILABLE, reason="iot_ingest_services not importable"
)


# ======================================================================
# Helpers
# ======================================================================

def _make_payload(sensor_id: int = 42, value: float = 23.5):
    """Create a mock MQTTReadingPayload."""
    payload = MagicMock()
    payload.sensor_id_int = sensor_id
    payload.sensor_id = str(sensor_id)
    payload.value = value
    payload.timestamp = "2026-01-01T00:00:00Z"
    payload.timestamp_float = 1735689600.0
    payload.sensor_type = "temperature"
    payload.model_dump.return_value = {
        "sensor_id": str(sensor_id), "value": value,
    }
    return payload


def _make_engine():
    """Create a mock SQLAlchemy engine."""
    engine = MagicMock()
    conn = MagicMock()
    engine.begin.return_value.__enter__ = MagicMock(return_value=conn)
    engine.begin.return_value.__exit__ = MagicMock(return_value=False)
    return engine


# ======================================================================
# Backward Compatibility (no CB)
# ======================================================================

class TestProcessorWithoutCB:

    def test_process_without_cb_success(self):
        engine = _make_engine()
        processor = ReadingProcessor(engine=engine)
        result = processor.process(_make_payload())
        assert result is True

    def test_process_without_cb_failure_raises(self):
        engine = _make_engine()
        engine.begin.return_value.__enter__.return_value.execute.side_effect = (
            Exception("SQL down")
        )
        processor = ReadingProcessor(engine=engine)
        with pytest.raises(Exception, match="SQL down"):
            processor.process(_make_payload())


# ======================================================================
# Circuit Breaker Integration
# ======================================================================

class TestProcessorWithCB:

    def _make_processor(self, engine=None, failing=False):
        engine = engine or _make_engine()
        if failing:
            engine.begin.return_value.__enter__.return_value.execute.side_effect = (
                Exception("SQL down")
            )
        cfg = CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout_seconds=0.5,
        )
        cb = CircuitBreaker("test_sp", config=cfg)
        dlq = MagicMock(spec=DeadLetterQueue)
        return ReadingProcessor(
            engine=engine, circuit_breaker=cb, dlq=dlq,
        ), cb, dlq

    def test_cb_success_path(self):
        proc, cb, dlq = self._make_processor()
        result = proc.process(_make_payload())
        assert result is True
        dlq.send.assert_not_called()
        assert cb.state == CircuitState.CLOSED

    def test_cb_opens_after_threshold(self):
        proc, cb, dlq = self._make_processor(failing=True)

        # 3 failures → circuit opens
        for _ in range(3):
            result = proc.process(_make_payload())
            assert result is False

        assert cb.state == CircuitState.OPEN

    def test_cb_open_sends_to_dlq(self):
        proc, cb, dlq = self._make_processor(failing=True)

        # Trip the breaker
        for _ in range(3):
            proc.process(_make_payload())

        # Next call should be rejected fast and go to DLQ
        dlq.send.reset_mock()
        result = proc.process(_make_payload(sensor_id=99))
        assert result is False
        dlq.send.assert_called_once()
        call_kwargs = dlq.send.call_args
        assert call_kwargs[1]["error_type"] == "sp_circuit_breaker"
        assert call_kwargs[1]["sensor_id"] == 99

    def test_cb_recovers_after_timeout(self):
        # Phase 1: failing engine to trip the breaker
        failing_engine = _make_engine()
        failing_engine.begin.return_value.__enter__.return_value.execute.side_effect = (
            Exception("SQL down")
        )
        cfg = CircuitBreakerConfig(
            failure_threshold=3, recovery_timeout_seconds=0.2,
            success_threshold=1,
        )
        cb = CircuitBreaker("test_recovery", config=cfg)
        dlq = MagicMock(spec=DeadLetterQueue)
        proc = ReadingProcessor(
            engine=failing_engine, circuit_breaker=cb, dlq=dlq,
        )

        # Trip the breaker
        for _ in range(3):
            proc.process(_make_payload())
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.3)

        # Phase 2: working engine — reuse same CB (now HALF_OPEN)
        ok_engine = _make_engine()
        proc2 = ReadingProcessor(
            engine=ok_engine, circuit_breaker=cb, dlq=dlq,
        )
        result = proc2.process(_make_payload())
        assert result is True
        assert cb.state == CircuitState.CLOSED

    def test_cb_failure_during_halfopen(self):
        engine = _make_engine()
        engine.begin.return_value.__enter__.return_value.execute.side_effect = (
            Exception("still down")
        )

        cfg = CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout_seconds=0.1,
        )
        cb = CircuitBreaker("test_halfopen", config=cfg)
        dlq = MagicMock(spec=DeadLetterQueue)
        proc = ReadingProcessor(engine=engine, circuit_breaker=cb, dlq=dlq)

        # Trip
        for _ in range(2):
            proc.process(_make_payload())
        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        time.sleep(0.15)

        # Attempt recovery — still failing → back to OPEN
        proc.process(_make_payload())
        assert cb.state == CircuitState.OPEN


# ======================================================================
# DLQ Unavailable
# ======================================================================

class TestProcessorDLQUnavailable:

    def test_no_dlq_graceful(self):
        """When DLQ is None, processor logs warning but doesn't crash."""
        engine = _make_engine()
        engine.begin.return_value.__enter__.return_value.execute.side_effect = (
            Exception("SQL down")
        )
        cfg = CircuitBreakerConfig(failure_threshold=1)
        cb = CircuitBreaker("test_no_dlq", config=cfg)
        proc = ReadingProcessor(engine=engine, circuit_breaker=cb, dlq=None)

        # Should not raise even though DLQ is None
        result = proc.process(_make_payload())
        assert result is False


# ======================================================================
# Feature Flags
# ======================================================================

class TestFeatureFlagDefaults:

    def test_sliding_window_flags_exist(self):
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        assert flags.ML_SLIDING_WINDOW_MAX_SENSORS == 1000
        assert flags.ML_SLIDING_WINDOW_TTL_SECONDS == 3600

    def test_circuit_breaker_flags_exist(self):
        from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
        flags = FeatureFlags()
        assert flags.ML_INGEST_CIRCUIT_BREAKER_ENABLED is True
        assert flags.ML_INGEST_CB_FAILURE_THRESHOLD == 5
        assert flags.ML_INGEST_CB_TIMEOUT_SECONDS == 30
