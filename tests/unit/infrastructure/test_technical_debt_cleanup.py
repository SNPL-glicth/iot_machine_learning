"""Tests for Phase 4 — Technical Debt Cleanup.

Covers:
- DEBT-1: safe_series_id_to_int() utility + all bridges use it
- DEBT-4: PredictionDomainService uses dataclasses.replace()
- COG-3: MetaDiagnostic deprecated, record_actual() decoupled
- COG-4: template_generator delegates severity to AnomalySeverity.from_score()
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from iot_machine_learning.domain.validators.input_guard import safe_series_id_to_int


# ── DEBT-1: safe_series_id_to_int ─────────────────────────────


class TestSafeSeriesIdToInt:

    def test_numeric_string(self) -> None:
        assert safe_series_id_to_int("42") == 42

    def test_zero_string(self) -> None:
        assert safe_series_id_to_int("0") == 0

    def test_large_number(self) -> None:
        assert safe_series_id_to_int("999999") == 999999

    def test_non_numeric_returns_fallback(self) -> None:
        assert safe_series_id_to_int("room_temp") == 0

    def test_non_numeric_custom_fallback(self) -> None:
        assert safe_series_id_to_int("abc", fallback=-1) == -1

    def test_empty_string_returns_fallback(self) -> None:
        assert safe_series_id_to_int("", fallback=0) == 0

    def test_negative_string(self) -> None:
        # "-5".isdigit() is False, but int("-5") works
        assert safe_series_id_to_int("-5") == -5

    def test_float_string_returns_fallback(self) -> None:
        # "3.14" is not convertible via int()
        assert safe_series_id_to_int("3.14") == 0

    def test_whitespace_returns_fallback(self) -> None:
        assert safe_series_id_to_int("  ") == 0


# ── DEBT-1: Bridges use safe_series_id_to_int ─────────────────


class TestBridgesUseSafeConversion:
    """Verify that domain port bridges no longer use bare int()."""

    def test_storage_port_bridge_non_numeric(self) -> None:
        """StoragePort.load_series_window handles non-numeric series_id."""
        from iot_machine_learning.domain.ports.storage_port import StoragePort
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorWindow,
        )

        class _StubStorage(StoragePort):
            def load_sensor_window(self, sensor_id, limit=500):
                return SensorWindow(sensor_id=sensor_id, readings=[])

            def save_prediction(self, prediction, **kw):
                return 1

            def save_anomaly_event(self, anomaly, **kw):
                return 1

            def list_active_sensor_ids(self):
                return []

            def get_latest_prediction(self, sensor_id):
                return None

            def get_sensor_metadata(self, sensor_id):
                return {"sensor_id": sensor_id}

            def get_device_id_for_sensor(self, sensor_id):
                return 0

        storage = _StubStorage()
        # Non-numeric series_id should NOT raise
        ts = storage.load_series_window("room_temp_1")
        assert ts is not None

    def test_audit_port_bridge_non_numeric(self) -> None:
        """AuditPort.log_series_prediction handles non-numeric series_id."""
        from iot_machine_learning.domain.ports.audit_port import AuditPort

        class _StubAudit(AuditPort):
            def __init__(self):
                self.logged_sensor_id = None

            def log_prediction(self, sensor_id, predicted_value,
                               confidence, engine_name, trace_id=None):
                self.logged_sensor_id = sensor_id

            def log_anomaly(self, sensor_id, value, score,
                            explanation, trace_id=None):
                pass

            def log_config_change(self, change_type, old_value,
                                  new_value, changed_by="system"):
                pass

            def log_event(self, event_type, data=None, trace_id=None):
                pass

        audit = _StubAudit()
        # Should not raise for non-numeric
        audit.log_series_prediction("device_abc", 25.0, 0.9, "taylor")
        assert audit.logged_sensor_id == 0  # fallback


# ── DEBT-4: dataclasses.replace() in PredictionDomainService ──


class TestDataclassesReplace:

    def test_replace_preserves_all_fields(self) -> None:
        """dataclasses.replace() on Prediction preserves all fields."""
        from iot_machine_learning.domain.entities.prediction import Prediction

        original = Prediction(
            series_id="42",
            predicted_value=25.0,
            confidence_score=0.9,
            trend="up",
            engine_name="taylor",
            horizon_steps=3,
            confidence_interval=(24.0, 26.0),
            feature_contributions={"slope": 0.8},
            metadata={"key": "value"},
            audit_trace_id=None,
        )
        updated = replace(original, audit_trace_id="trace_123")

        # Changed field
        assert updated.audit_trace_id == "trace_123"
        # All other fields preserved
        assert updated.series_id == "42"
        assert updated.predicted_value == 25.0
        assert updated.confidence_score == 0.9
        assert updated.trend == "up"
        assert updated.engine_name == "taylor"
        assert updated.horizon_steps == 3
        assert updated.confidence_interval == (24.0, 26.0)
        assert updated.feature_contributions == {"slope": 0.8}
        assert updated.metadata == {"key": "value"}

    def test_replace_multiple_fields(self) -> None:
        from iot_machine_learning.domain.entities.prediction import Prediction

        original = Prediction(
            series_id="1",
            predicted_value=10.0,
            confidence_score=0.5,
            trend="stable",
            engine_name="baseline",
        )
        updated = replace(
            original,
            engine_name="baseline_fallback",
            metadata={"fallback_reason": "error"},
            audit_trace_id="abc",
        )
        assert updated.engine_name == "baseline_fallback"
        assert updated.metadata == {"fallback_reason": "error"}
        assert updated.audit_trace_id == "abc"
        # Unchanged
        assert updated.predicted_value == 10.0

    def test_prediction_domain_service_sets_trace_id(self) -> None:
        """PredictionDomainService enriches prediction with trace_id."""
        from iot_machine_learning.domain.entities.prediction import Prediction
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading,
            SensorWindow,
        )
        from iot_machine_learning.domain.ports.prediction_port import PredictionPort
        from iot_machine_learning.domain.services.prediction_domain_service import (
            PredictionDomainService,
        )

        class _StubPort(PredictionPort):
            @property
            def name(self) -> str:
                return "stub"

            def can_handle(self, n_points: int) -> bool:
                return True

            def predict(self, window):
                return Prediction(
                    series_id=str(window.sensor_id),
                    predicted_value=42.0,
                    confidence_score=0.9,
                    trend="stable",
                    engine_name="stub",
                )

        svc = PredictionDomainService(engines=[_StubPort()])
        window = SensorWindow(
            sensor_id=1,
            readings=[SensorReading(sensor_id=1, value=20.0, timestamp=1.0)],
        )
        result = svc.predict(window)
        assert result.audit_trace_id is not None
        assert len(result.audit_trace_id) == 12


# ── COG-3: MetaDiagnostic deprecation ─────────────────────────


class TestMetaDiagnosticDeprecation:

    def test_last_diagnostic_emits_deprecation_warning(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestrator import (
            MetaCognitiveOrchestrator,
        )
        from iot_machine_learning.infrastructure.ml.interfaces import (
            PredictionEngine,
            PredictionResult,
        )

        class _Stub(PredictionEngine):
            @property
            def name(self) -> str:
                return "s"

            def can_handle(self, n_points: int) -> bool:
                return True

            def predict(self, values, timestamps=None):
                return PredictionResult(
                    predicted_value=1.0, confidence=0.8, trend="stable",
                )

        orch = MetaCognitiveOrchestrator([_Stub()], enable_plasticity=False)
        orch.predict([1.0, 2.0, 3.0])

        with pytest.warns(DeprecationWarning, match="last_diagnostic"):
            _ = orch.last_diagnostic

    def test_last_explanation_no_warning(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestrator import (
            MetaCognitiveOrchestrator,
        )
        from iot_machine_learning.infrastructure.ml.interfaces import (
            PredictionEngine,
            PredictionResult,
        )

        class _Stub(PredictionEngine):
            @property
            def name(self) -> str:
                return "s"

            def can_handle(self, n_points: int) -> bool:
                return True

            def predict(self, values, timestamps=None):
                return PredictionResult(
                    predicted_value=1.0, confidence=0.8, trend="stable",
                )

        orch = MetaCognitiveOrchestrator([_Stub()], enable_plasticity=False)
        orch.predict([1.0, 2.0, 3.0])

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            # Should NOT raise
            expl = orch.last_explanation
            assert expl is not None

    def test_record_actual_works_without_diagnostic(self) -> None:
        """record_actual uses internal state, not MetaDiagnostic."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestrator import (
            MetaCognitiveOrchestrator,
        )
        from iot_machine_learning.infrastructure.ml.interfaces import (
            PredictionEngine,
            PredictionResult,
        )

        class _Stub(PredictionEngine):
            @property
            def name(self) -> str:
                return "s"

            def can_handle(self, n_points: int) -> bool:
                return True

            def predict(self, values, timestamps=None):
                return PredictionResult(
                    predicted_value=10.0, confidence=0.8, trend="stable",
                )

        orch = MetaCognitiveOrchestrator([_Stub()], enable_plasticity=True)
        orch.predict([1.0, 2.0, 3.0])
        # Should work without accessing last_diagnostic
        orch.record_actual(10.5)  # no crash

    def test_record_actual_before_predict_is_noop(self) -> None:
        from iot_machine_learning.infrastructure.ml.cognitive.orchestrator import (
            MetaCognitiveOrchestrator,
        )
        from iot_machine_learning.infrastructure.ml.interfaces import (
            PredictionEngine,
            PredictionResult,
        )

        class _Stub(PredictionEngine):
            @property
            def name(self) -> str:
                return "s"

            def can_handle(self, n_points: int) -> bool:
                return True

            def predict(self, values, timestamps=None):
                return PredictionResult(
                    predicted_value=1.0, confidence=0.8, trend="stable",
                )

        orch = MetaCognitiveOrchestrator([_Stub()], enable_plasticity=True)
        # Before any predict — should be a no-op
        orch.record_actual(5.0)  # no crash


# ── Verify no remaining bare int(series_id) ───────────────


class TestNoRemainingBareIntConversions:
    """Meta-test: grep for remaining unsafe patterns in domain ports."""

    def test_no_bare_int_series_id_in_ports(self) -> None:
        """Ensure domain ports don't have bare int(series_id) anymore."""
        import inspect
        import re
        from iot_machine_learning.domain.ports import (
            audit_port,
            prediction_port,
            anomaly_detection_port,
            pattern_detection_port,
            storage_port,
        )

        # Match bare `int(series_id)` or `int(series.series_id)` but NOT
        # `safe_series_id_to_int(series_id)` which is the safe replacement.
        bare_pattern = re.compile(
            r'(?<!safe_series_id_to_)int\((series_id|series\.series_id)\)'
        )

        for mod in [
            audit_port,
            prediction_port,
            anomaly_detection_port,
            pattern_detection_port,
            storage_port,
        ]:
            source = inspect.getsource(mod)
            matches = bare_pattern.findall(source)
            assert not matches, (
                f"{mod.__name__} still has bare int() conversion: {matches}"
            )
