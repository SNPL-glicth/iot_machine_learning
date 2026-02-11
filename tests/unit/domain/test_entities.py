"""Tests para entidades del dominio.

Verifica:
- SensorReading: validación de valores, propiedades.
- SensorWindow: acceso a valores, timestamps, propiedades.
- Prediction: confidence levels, audit dict.
- AnomalyResult: severity from score, factory normal.
- Pattern entities: ChangePoint, DeltaSpikeResult, OperationalRegime.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.entities.sensor_reading import (
    SensorReading,
    SensorWindow,
)
from iot_machine_learning.domain.entities.prediction import (
    Prediction,
    PredictionConfidence,
)
from iot_machine_learning.domain.entities.anomaly import (
    AnomalyResult,
    AnomalySeverity,
)
from iot_machine_learning.domain.entities.pattern import (
    ChangePoint,
    ChangePointType,
    DeltaSpikeResult,
    OperationalRegime,
    PatternResult,
    PatternType,
    SpikeClassification,
)


class TestSensorReading:
    """Tests para SensorReading value object."""

    def test_valid_reading(self) -> None:
        r = SensorReading(sensor_id=1, value=22.5, timestamp=1000.0)
        assert r.sensor_id == 1
        assert r.value == 22.5
        assert r.is_valid is True

    def test_nan_value_raises(self) -> None:
        with pytest.raises(ValueError, match="finito"):
            SensorReading(sensor_id=1, value=float("nan"), timestamp=1000.0)

    def test_inf_value_raises(self) -> None:
        with pytest.raises(ValueError, match="finito"):
            SensorReading(sensor_id=1, value=float("inf"), timestamp=1000.0)

    def test_frozen(self) -> None:
        r = SensorReading(sensor_id=1, value=22.5, timestamp=1000.0)
        with pytest.raises(AttributeError):
            r.value = 30.0  # type: ignore[misc]


class TestSensorWindow:
    """Tests para SensorWindow value object."""

    def test_empty_window(self) -> None:
        w = SensorWindow(sensor_id=1)
        assert w.is_empty is True
        assert w.size == 0
        assert w.values == []
        assert w.last_value is None

    def test_window_with_readings(self) -> None:
        readings = [
            SensorReading(sensor_id=1, value=20.0, timestamp=100.0),
            SensorReading(sensor_id=1, value=21.0, timestamp=101.0),
            SensorReading(sensor_id=1, value=22.0, timestamp=102.0),
        ]
        w = SensorWindow(sensor_id=1, readings=readings)

        assert w.size == 3
        assert w.values == [20.0, 21.0, 22.0]
        assert w.timestamps == [100.0, 101.0, 102.0]
        assert w.last_value == 22.0
        assert w.last_timestamp == 102.0
        assert w.time_span_seconds == 2.0


class TestPrediction:
    """Tests para Prediction entity."""

    def test_confidence_levels(self) -> None:
        assert PredictionConfidence.from_score(0.1) == PredictionConfidence.VERY_LOW
        assert PredictionConfidence.from_score(0.3) == PredictionConfidence.LOW
        assert PredictionConfidence.from_score(0.5) == PredictionConfidence.MEDIUM
        assert PredictionConfidence.from_score(0.8) == PredictionConfidence.HIGH
        assert PredictionConfidence.from_score(0.95) == PredictionConfidence.VERY_HIGH

    def test_prediction_audit_dict(self) -> None:
        p = Prediction(
            sensor_id=1,
            predicted_value=22.5,
            confidence_score=0.85,
            trend="up",
            engine_name="taylor",
            audit_trace_id="abc123",
        )
        d = p.to_audit_dict()
        assert d["sensor_id"] == 1
        assert d["predicted_value"] == 22.5
        assert d["confidence_level"] == "high"
        assert d["audit_trace_id"] == "abc123"

    def test_has_confidence_interval(self) -> None:
        p1 = Prediction(
            sensor_id=1, predicted_value=22.5,
            confidence_score=0.8, trend="stable", engine_name="test",
        )
        assert p1.has_confidence_interval is False

        p2 = Prediction(
            sensor_id=1, predicted_value=22.5,
            confidence_score=0.8, trend="stable", engine_name="test",
            confidence_interval=(21.0, 24.0),
        )
        assert p2.has_confidence_interval is True


class TestAnomalyResult:
    """Tests para AnomalyResult entity."""

    def test_severity_from_score(self) -> None:
        assert AnomalySeverity.from_score(0.1) == AnomalySeverity.NONE
        assert AnomalySeverity.from_score(0.4) == AnomalySeverity.LOW
        assert AnomalySeverity.from_score(0.6) == AnomalySeverity.MEDIUM
        assert AnomalySeverity.from_score(0.8) == AnomalySeverity.HIGH
        assert AnomalySeverity.from_score(0.95) == AnomalySeverity.CRITICAL

    def test_normal_factory(self) -> None:
        r = AnomalyResult.normal(sensor_id=42)
        assert r.is_anomaly is False
        assert r.score == 0.0
        assert r.severity == AnomalySeverity.NONE
        assert r.sensor_id == 42

    def test_audit_dict(self) -> None:
        r = AnomalyResult(
            sensor_id=1, is_anomaly=True, score=0.85,
            method_votes={"z_score": 0.9, "iqr": 0.8},
            explanation="Z-score alto",
            severity=AnomalySeverity.HIGH,
        )
        d = r.to_audit_dict()
        assert d["is_anomaly"] is True
        assert d["severity"] == "high"


class TestPatternEntities:
    """Tests para entidades de patrones."""

    def test_pattern_result(self) -> None:
        p = PatternResult(
            sensor_id=1,
            pattern_type=PatternType.SPIKE,
            confidence=0.9,
            description="Spike detectado",
        )
        assert p.pattern_type == PatternType.SPIKE

    def test_change_point(self) -> None:
        cp = ChangePoint(
            index=50,
            change_type=ChangePointType.LEVEL_SHIFT,
            magnitude=5.0,
            confidence=0.85,
            before_mean=20.0,
            after_mean=25.0,
        )
        assert cp.change_type == ChangePointType.LEVEL_SHIFT
        assert cp.magnitude == 5.0

    def test_delta_spike_result(self) -> None:
        ds = DeltaSpikeResult(
            is_delta_spike=True,
            confidence=0.8,
            delta_magnitude=10.0,
            persistence_score=0.9,
            classification=SpikeClassification.DELTA_SPIKE,
            explanation="Cambio legítimo",
        )
        assert ds.is_delta_spike is True
        assert ds.classification == SpikeClassification.DELTA_SPIKE

    def test_operational_regime(self) -> None:
        r = OperationalRegime(
            regime_id=0,
            name="idle",
            mean_value=22.0,
            std_value=0.5,
        )
        assert r.name == "idle"
        assert r.mean_value == 22.0
