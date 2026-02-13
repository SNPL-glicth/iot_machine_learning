"""Tests para migración de ports agnósticos (series_id: str).

Cubre:
- StoragePort: default bridge methods (load_series_window, list_active_series_ids,
  get_latest_prediction_for_series, get_series_metadata)
- AuditPort: default bridge methods (log_series_prediction, log_series_anomaly)
- FileAuditLogger: log_series_prediction, log_series_anomaly escriben correctamente
- NullAuditLogger: nuevos métodos no crashean
- CognitiveStorageDecorator: delega nuevos métodos al inner
- Domain services: usan series_id audit methods
- Backward compatibility: legacy sensor_id methods siguen funcionando
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from iot_machine_learning.domain.entities.anomaly import AnomalyResult, AnomalySeverity
from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import (
    SensorReading,
    SensorWindow,
)
from iot_machine_learning.domain.entities.time_series import TimeSeries
from iot_machine_learning.domain.ports.audit_port import AuditPort
from iot_machine_learning.domain.ports.storage_port import StoragePort


# ── Concrete test implementations ─────────────────────────────────────


class FakeStorage(StoragePort):
    """Minimal concrete StoragePort for testing default bridge methods."""

    def __init__(self) -> None:
        self.loaded_sensor_ids: list[int] = []
        self.listed: bool = False
        self.saved_predictions: list[Prediction] = []
        self.latest_predictions: dict[int, Prediction] = {}
        self.saved_anomalies: list[AnomalyResult] = []
        self.metadata: dict[int, dict] = {}

    def load_sensor_window(self, sensor_id: int, limit: int = 500) -> SensorWindow:
        self.loaded_sensor_ids.append(sensor_id)
        readings = [
            SensorReading(sensor_id=sensor_id, value=float(i), timestamp=float(i))
            for i in range(min(limit, 5))
        ]
        return SensorWindow(sensor_id=sensor_id, readings=readings)

    def list_active_sensor_ids(self) -> List[int]:
        self.listed = True
        return [1, 2, 3]

    def save_prediction(self, prediction: Prediction) -> int:
        self.saved_predictions.append(prediction)
        return len(self.saved_predictions)

    def get_latest_prediction(self, sensor_id: int) -> Optional[Prediction]:
        return self.latest_predictions.get(sensor_id)

    def save_anomaly_event(
        self, anomaly: AnomalyResult, prediction_id: Optional[int] = None
    ) -> int:
        self.saved_anomalies.append(anomaly)
        return len(self.saved_anomalies)

    def get_sensor_metadata(self, sensor_id: int) -> Dict[str, object]:
        return self.metadata.get(sensor_id, {"sensor_id": sensor_id})

    def get_device_id_for_sensor(self, sensor_id: int) -> int:
        return sensor_id * 10


class FakeAudit(AuditPort):
    """Minimal concrete AuditPort for testing default bridge methods."""

    def __init__(self) -> None:
        self.events: list[dict] = []
        self.predictions: list[dict] = []
        self.anomalies: list[dict] = []

    def log_event(self, event_type: str, action: str, resource: str,
                  details: Dict[str, Any], result: str = "success",
                  user_id: Optional[str] = None,
                  before_state: Optional[Dict[str, Any]] = None,
                  after_state: Optional[Dict[str, Any]] = None) -> None:
        self.events.append({
            "event_type": event_type, "action": action,
            "resource": resource, "details": details,
        })

    def log_prediction(self, sensor_id: int, predicted_value: float,
                       confidence: float, engine_name: str,
                       trace_id: Optional[str] = None) -> None:
        self.predictions.append({
            "sensor_id": sensor_id, "predicted_value": predicted_value,
            "confidence": confidence, "engine_name": engine_name,
            "trace_id": trace_id,
        })

    def log_anomaly(self, sensor_id: int, value: float, score: float,
                    explanation: str, trace_id: Optional[str] = None) -> None:
        self.anomalies.append({
            "sensor_id": sensor_id, "value": value, "score": score,
            "explanation": explanation, "trace_id": trace_id,
        })

    def log_config_change(self, config_key: str, old_value: Any,
                          new_value: Any, user_id: str) -> None:
        pass


# ── StoragePort default bridges ───────────────────────────────────────


class TestStoragePortBridges:
    """Tests para default bridge methods de StoragePort."""

    def test_load_series_window_delegates_to_sensor(self):
        storage = FakeStorage()
        ts = storage.load_series_window("42", limit=5)

        assert isinstance(ts, TimeSeries)
        assert ts.series_id == "42"
        assert 42 in storage.loaded_sensor_ids

    def test_load_series_window_non_numeric_id(self):
        storage = FakeStorage()
        ts = storage.load_series_window("temperature_room_1")

        assert isinstance(ts, TimeSeries)
        # Non-numeric → sensor_id=0
        assert 0 in storage.loaded_sensor_ids

    def test_list_active_series_ids(self):
        storage = FakeStorage()
        ids = storage.list_active_series_ids()

        assert ids == ["1", "2", "3"]
        assert storage.listed is True

    def test_get_latest_prediction_for_series(self):
        storage = FakeStorage()
        pred = Prediction(
            series_id="10", predicted_value=22.5,
            confidence_score=0.9, trend="up", engine_name="taylor",
        )
        storage.latest_predictions[10] = pred

        result = storage.get_latest_prediction_for_series("10")
        assert result is not None
        assert result.predicted_value == 22.5

    def test_get_latest_prediction_for_series_not_found(self):
        storage = FakeStorage()
        result = storage.get_latest_prediction_for_series("999")
        assert result is None

    def test_get_series_metadata(self):
        storage = FakeStorage()
        storage.metadata[5] = {"sensor_id": 5, "name": "temp", "unit": "°C"}

        meta = storage.get_series_metadata("5")
        assert meta["name"] == "temp"
        assert meta["unit"] == "°C"

    def test_get_series_metadata_non_numeric(self):
        storage = FakeStorage()
        meta = storage.get_series_metadata("room_temp")
        # Falls back to sensor_id=0
        assert meta["sensor_id"] == 0


# ── AuditPort default bridges ─────────────────────────────────────────


class TestAuditPortBridges:
    """Tests para default bridge methods de AuditPort."""

    def test_log_series_prediction_delegates(self):
        audit = FakeAudit()
        audit.log_series_prediction(
            series_id="42",
            predicted_value=22.5,
            confidence=0.85,
            engine_name="taylor",
            trace_id="abc",
        )

        assert len(audit.predictions) == 1
        assert audit.predictions[0]["sensor_id"] == 42
        assert audit.predictions[0]["predicted_value"] == 22.5
        assert audit.predictions[0]["trace_id"] == "abc"

    def test_log_series_prediction_non_numeric(self):
        audit = FakeAudit()
        audit.log_series_prediction(
            series_id="room_temp",
            predicted_value=22.5,
            confidence=0.85,
            engine_name="taylor",
        )

        assert audit.predictions[0]["sensor_id"] == 0

    def test_log_series_anomaly_delegates(self):
        audit = FakeAudit()
        audit.log_series_anomaly(
            series_id="7",
            value=50.0,
            score=0.9,
            explanation="Z-score alto",
            trace_id="xyz",
        )

        assert len(audit.anomalies) == 1
        assert audit.anomalies[0]["sensor_id"] == 7
        assert audit.anomalies[0]["score"] == 0.9
        assert audit.anomalies[0]["trace_id"] == "xyz"

    def test_log_series_anomaly_non_numeric(self):
        audit = FakeAudit()
        audit.log_series_anomaly(
            series_id="vibration_motor_3",
            value=50.0,
            score=0.9,
            explanation="test",
        )

        assert audit.anomalies[0]["sensor_id"] == 0


# ── FileAuditLogger series methods ────────────────────────────────────


class TestFileAuditLoggerSeries:
    """Tests para FileAuditLogger con métodos series_id."""

    @pytest.fixture
    def audit_file(self, tmp_path: Path) -> Path:
        return tmp_path / "audit_series.jsonl"

    def test_log_series_prediction_writes_event(self, audit_file: Path):
        from iot_machine_learning.infrastructure.security.audit_logger import (
            FileAuditLogger,
        )

        logger = FileAuditLogger(log_file=audit_file, include_hash=False)
        logger.log_series_prediction(
            series_id="42",
            predicted_value=22.5,
            confidence=0.85,
            engine_name="taylor",
            trace_id="trace_1",
        )

        entry = json.loads(audit_file.read_text().strip())
        assert entry["event_type"] == "prediction"
        assert entry["resource"] == "sensor_42"
        assert entry["details"]["engine"] == "taylor"

    def test_log_series_anomaly_writes_event(self, audit_file: Path):
        from iot_machine_learning.infrastructure.security.audit_logger import (
            FileAuditLogger,
        )

        logger = FileAuditLogger(log_file=audit_file, include_hash=False)
        logger.log_series_anomaly(
            series_id="7",
            value=50.0,
            score=0.9,
            explanation="Spike detectado",
            trace_id="trace_2",
        )

        entry = json.loads(audit_file.read_text().strip())
        assert entry["event_type"] == "anomaly_detection"
        assert entry["details"]["anomaly_score"] == 0.9

    def test_log_series_prediction_non_numeric_id(self, audit_file: Path):
        from iot_machine_learning.infrastructure.security.audit_logger import (
            FileAuditLogger,
        )

        logger = FileAuditLogger(log_file=audit_file, include_hash=False)
        logger.log_series_prediction(
            series_id="vibration_motor",
            predicted_value=1.5,
            confidence=0.7,
            engine_name="kalman",
        )

        entry = json.loads(audit_file.read_text().strip())
        # Non-numeric series_id → sensor_0
        assert entry["resource"] == "sensor_0"


# ── NullAuditLogger series methods ────────────────────────────────────


class TestNullAuditLoggerSeries:
    """NullAuditLogger no debe crashear con nuevos métodos."""

    def test_no_crash_on_series_methods(self):
        from iot_machine_learning.infrastructure.security.audit_logger import (
            NullAuditLogger,
        )

        logger = NullAuditLogger()
        logger.log_series_prediction("42", 22.5, 0.85, "taylor")
        logger.log_series_anomaly("7", 50.0, 0.9, "test")


# ── Domain services use series_id ─────────────────────────────────────


class TestDomainServicesUseSeries:
    """Verifica que domain services usan log_series_* en vez de log_*."""

    def test_prediction_service_uses_log_series_prediction(self):
        """PredictionDomainService debe llamar log_series_prediction."""
        audit = FakeAudit()
        # Patch log_series_prediction to track calls
        calls = []
        original = audit.log_series_prediction

        def tracking_log(*args, **kwargs):
            calls.append(kwargs or dict(zip(
                ["series_id", "predicted_value", "confidence", "engine_name", "trace_id"],
                args,
            )))
            return original(*args, **kwargs)

        audit.log_series_prediction = tracking_log

        from iot_machine_learning.domain.services.prediction_domain_service import (
            PredictionDomainService,
        )
        from iot_machine_learning.domain.ports.prediction_port import PredictionPort

        # Create a mock engine
        mock_engine = MagicMock(spec=PredictionPort)
        mock_engine.name = "test_engine"
        mock_engine.can_handle.return_value = True
        mock_engine.predict.return_value = Prediction(
            series_id="42",
            predicted_value=22.5,
            confidence_score=0.85,
            trend="up",
            engine_name="test_engine",
        )

        service = PredictionDomainService(
            engines=[mock_engine],
            audit=audit,
        )

        window = SensorWindow(
            sensor_id=42,
            readings=[
                SensorReading(sensor_id=42, value=float(i), timestamp=float(i))
                for i in range(10)
            ],
        )
        service.predict(window)

        # Should have called log_series_prediction, not log_prediction
        assert len(calls) == 1
        assert calls[0]["series_id"] == "42"

    def test_anomaly_service_uses_log_series_anomaly(self):
        """AnomalyDomainService debe llamar log_series_anomaly."""
        audit = FakeAudit()
        calls = []
        original = audit.log_series_anomaly

        def tracking_log(*args, **kwargs):
            calls.append(kwargs or {})
            return original(*args, **kwargs)

        audit.log_series_anomaly = tracking_log

        from iot_machine_learning.domain.services.anomaly_domain_service import (
            AnomalyDomainService,
        )
        from iot_machine_learning.domain.ports.anomaly_detection_port import (
            AnomalyDetectionPort,
        )

        # Create a mock detector that returns anomaly
        mock_detector = MagicMock(spec=AnomalyDetectionPort)
        mock_detector.name = "test_detector"
        mock_detector.is_trained.return_value = True
        mock_detector.detect.return_value = AnomalyResult(
            series_id="7",
            is_anomaly=True,
            score=0.9,
            method_votes={"test": 0.9},
            confidence=0.85,
            explanation="Test anomaly",
            severity=AnomalySeverity.HIGH,
        )

        service = AnomalyDomainService(
            detectors=[mock_detector],
            audit=audit,
        )

        window = SensorWindow(
            sensor_id=7,
            readings=[
                SensorReading(sensor_id=7, value=50.0, timestamp=float(i))
                for i in range(5)
            ],
        )
        service.detect(window)

        assert len(calls) == 1
        assert calls[0]["series_id"] == "7"


# ── CognitiveStorageDecorator series methods ──────────────────────────


class TestCognitiveStorageDecoratorSeries:
    """Tests para CognitiveStorageDecorator con métodos series_id."""

    def test_load_series_window_delegates(self):
        from iot_machine_learning.infrastructure.adapters.cognitive_storage_decorator import (
            CognitiveStorageDecorator,
        )

        inner = FakeStorage()
        cognitive = MagicMock()
        flags = MagicMock()
        flags.ML_ENABLE_COGNITIVE_MEMORY = False

        decorator = CognitiveStorageDecorator(inner, cognitive, flags)
        ts = decorator.load_series_window("42", limit=5)

        assert isinstance(ts, TimeSeries)
        assert 42 in inner.loaded_sensor_ids

    def test_list_active_series_ids_delegates(self):
        from iot_machine_learning.infrastructure.adapters.cognitive_storage_decorator import (
            CognitiveStorageDecorator,
        )

        inner = FakeStorage()
        cognitive = MagicMock()
        flags = MagicMock()
        flags.ML_ENABLE_COGNITIVE_MEMORY = False

        decorator = CognitiveStorageDecorator(inner, cognitive, flags)
        ids = decorator.list_active_series_ids()

        assert ids == ["1", "2", "3"]

    def test_get_series_metadata_delegates(self):
        from iot_machine_learning.infrastructure.adapters.cognitive_storage_decorator import (
            CognitiveStorageDecorator,
        )

        inner = FakeStorage()
        inner.metadata[5] = {"sensor_id": 5, "name": "temp"}
        cognitive = MagicMock()
        flags = MagicMock()
        flags.ML_ENABLE_COGNITIVE_MEMORY = False

        decorator = CognitiveStorageDecorator(inner, cognitive, flags)
        meta = decorator.get_series_metadata("5")

        assert meta["name"] == "temp"
