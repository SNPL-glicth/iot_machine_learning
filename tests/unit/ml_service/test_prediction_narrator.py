"""Tests para PredictionNarrator y funciones de narrativa.

Verifica lógica de Narrative pura — sin BD, sin I/O.
"""

from __future__ import annotations

import json
import pytest
from dataclasses import dataclass
from unittest.mock import MagicMock

from iot_machine_learning.ml_service.runners.common.prediction_narrator import (
    PredictionNarrator,
    build_short_message,
    build_explanation_payload,
)


# --- Fakes ---

@dataclass
class FakeSensorMetadata:
    sensor_id: int = 1
    sensor_type: str = "temperature"
    unit: str = "°C"
    location: str = "Sala de servidores"
    criticality: str = "high"


@dataclass
class FakeSeverityResult:
    risk_level: str = "LOW"
    severity: str = "info"
    action_required: bool = False
    recommended_action: str = "Sin acción requerida."


class FakeSeverityClassifier:
    """Fake SeverityClassifier que retorna resultado configurable."""

    def __init__(self, result: FakeSeverityResult | None = None):
        self._result = result or FakeSeverityResult()

    def classify(self, **kwargs) -> FakeSeverityResult:
        return self._result


# --- Tests para build_short_message ---

class TestBuildShortMessage:
    """Tests para generación de mensaje corto."""

    def test_critical_message(self) -> None:
        msg = build_short_message("critical", "temperature", "Sala A")
        assert "Riesgo crítico" in msg
        assert "temperature" in msg
        assert "Sala A" in msg

    def test_warning_message(self) -> None:
        msg = build_short_message("warning", "humidity", "Bodega")
        assert "Comportamiento inusual" in msg
        assert "humidity" in msg
        assert "Bodega" in msg

    def test_info_message(self) -> None:
        msg = build_short_message("info", "power", "Planta 1")
        assert "Predicción estable" in msg
        assert "power" in msg
        assert "Planta 1" in msg

    def test_unknown_severity_defaults_to_stable(self) -> None:
        msg = build_short_message("unknown", "voltage", "Lab")
        assert "Predicción estable" in msg


# --- Tests para build_explanation_payload ---

class TestBuildExplanationPayload:
    """Tests para construcción de JSON de explicación."""

    def test_returns_valid_json(self) -> None:
        result = build_explanation_payload(
            severity="warning",
            short_message="Test message",
            recommended_action="Check sensor",
            predicted_value=25.5,
            trend="up",
            anomaly_score=0.7,
            confidence=0.85,
            horizon_minutes=10,
            risk_level="MEDIUM",
            sensor_type="temperature",
            location="Lab",
        )

        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    def test_severity_is_uppercase(self) -> None:
        result = build_explanation_payload(
            severity="warning",
            short_message="msg",
            recommended_action="act",
            predicted_value=25.0,
            trend="stable",
            anomaly_score=0.0,
            confidence=0.5,
            horizon_minutes=10,
            risk_level="LOW",
            sensor_type="temp",
            location="loc",
        )

        parsed = json.loads(result)
        assert parsed["severity"] == "WARNING"

    def test_details_contains_all_fields(self) -> None:
        result = build_explanation_payload(
            severity="info",
            short_message="msg",
            recommended_action="act",
            predicted_value=22.0,
            trend="down",
            anomaly_score=0.1,
            confidence=0.9,
            horizon_minutes=15,
            risk_level="HIGH",
            sensor_type="humidity",
            location="Warehouse",
        )

        parsed = json.loads(result)
        details = parsed["details"]
        assert details["predicted_value"] == 22.0
        assert details["trend"] == "down"
        assert details["anomaly_score"] == 0.1
        assert details["confidence"] == 0.9
        assert details["horizon_minutes"] == 15
        assert details["risk_level"] == "HIGH"
        assert details["sensor_type"] == "humidity"
        assert details["location"] == "Warehouse"

    def test_source_is_ml_baseline(self) -> None:
        result = build_explanation_payload(
            severity="info",
            short_message="msg",
            recommended_action="act",
            predicted_value=0.0,
            trend="stable",
            anomaly_score=0.0,
            confidence=0.5,
            horizon_minutes=10,
            risk_level="NONE",
            sensor_type="t",
            location="l",
        )

        parsed = json.loads(result)
        assert parsed["source"] == "ml_baseline"


# --- Tests para PredictionNarrator ---

class TestPredictionNarrator:
    """Tests para el narrador completo."""

    def test_build_explanation_returns_prediction_explanation(self) -> None:
        classifier = FakeSeverityClassifier()
        narrator = PredictionNarrator(classifier)
        meta = FakeSensorMetadata()

        result = narrator.build_explanation(
            sensor_meta=meta,
            predicted_value=25.0,
            trend="stable",
            anomaly=False,
            anomaly_score=0.0,
            confidence=0.8,
            horizon_minutes=10,
        )

        assert result.sensor_id == 1
        assert result.predicted_value == 25.0
        assert result.trend == "stable"
        assert result.anomaly is False

    def test_critical_severity_forces_anomaly_score(self) -> None:
        """Si severity=critical pero anomaly_score<=0, se fuerza a 0.5."""
        sev_result = FakeSeverityResult(severity="critical", risk_level="HIGH")
        classifier = FakeSeverityClassifier(sev_result)
        narrator = PredictionNarrator(classifier)
        meta = FakeSensorMetadata()

        result = narrator.build_explanation(
            sensor_meta=meta,
            predicted_value=50.0,
            trend="up",
            anomaly=True,
            anomaly_score=0.0,  # <= 0
            confidence=0.9,
            horizon_minutes=10,
        )

        assert result.anomaly_score == 0.5

    def test_normal_anomaly_score_not_modified(self) -> None:
        sev_result = FakeSeverityResult(severity="warning")
        classifier = FakeSeverityClassifier(sev_result)
        narrator = PredictionNarrator(classifier)
        meta = FakeSensorMetadata()

        result = narrator.build_explanation(
            sensor_meta=meta,
            predicted_value=30.0,
            trend="up",
            anomaly=True,
            anomaly_score=0.75,
            confidence=0.8,
            horizon_minutes=10,
        )

        assert result.anomaly_score == 0.75

    def test_explanation_is_valid_json(self) -> None:
        classifier = FakeSeverityClassifier()
        narrator = PredictionNarrator(classifier)
        meta = FakeSensorMetadata()

        result = narrator.build_explanation(
            sensor_meta=meta,
            predicted_value=22.0,
            trend="stable",
            anomaly=False,
            anomaly_score=0.0,
            confidence=0.7,
            horizon_minutes=10,
        )

        parsed = json.loads(result.explanation)
        assert "severity" in parsed
        assert "short_message" in parsed
        assert "details" in parsed

    def test_severity_and_risk_level_propagated(self) -> None:
        sev_result = FakeSeverityResult(
            severity="warning",
            risk_level="MEDIUM",
            action_required=True,
            recommended_action="Revisar equipo",
        )
        classifier = FakeSeverityClassifier(sev_result)
        narrator = PredictionNarrator(classifier)
        meta = FakeSensorMetadata()

        result = narrator.build_explanation(
            sensor_meta=meta,
            predicted_value=35.0,
            trend="up",
            anomaly=True,
            anomaly_score=0.6,
            confidence=0.8,
            horizon_minutes=10,
        )

        assert result.severity == "warning"
        assert result.risk_level == "MEDIUM"
        assert result.action_required is True
        assert result.recommended_action == "Revisar equipo"

    def test_user_defined_range_passed_to_classifier(self) -> None:
        """Verifica que user_defined_range se pasa al clasificador."""
        mock_classifier = MagicMock()
        mock_classifier.classify.return_value = FakeSeverityResult()
        narrator = PredictionNarrator(mock_classifier)
        meta = FakeSensorMetadata()

        narrator.build_explanation(
            sensor_meta=meta,
            predicted_value=25.0,
            trend="stable",
            anomaly=False,
            anomaly_score=0.0,
            confidence=0.8,
            horizon_minutes=10,
            user_defined_range=(20.0, 30.0),
        )

        mock_classifier.classify.assert_called_once()
        call_kwargs = mock_classifier.classify.call_args[1]
        assert call_kwargs["user_defined_range"] == (20.0, 30.0)
