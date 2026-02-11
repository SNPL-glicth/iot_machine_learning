"""Tests para domain/services/severity_rules.py.

Verifica reglas de negocio PURAS — sin I/O, sin mocks de BD.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.services.severity_rules import (
    SeverityResult,
    build_recommended_action,
    classify_severity,
    compute_risk_level,
    compute_severity,
    is_out_of_range,
)
from iot_machine_learning.domain.entities.sensor_ranges import (
    DEFAULT_SENSOR_RANGES,
    get_default_range,
)


# --- sensor_ranges ---

class TestSensorRanges:
    """Tests para rangos de sensor de dominio."""

    def test_temperature_range_exists(self) -> None:
        rng = get_default_range("temperature")
        assert rng == (15.0, 35.0)

    def test_humidity_range_exists(self) -> None:
        rng = get_default_range("humidity")
        assert rng == (30.0, 70.0)

    def test_unknown_type_returns_none(self) -> None:
        assert get_default_range("unknown_sensor") is None

    def test_all_ranges_are_tuples(self) -> None:
        for sensor_type, rng in DEFAULT_SENSOR_RANGES.items():
            assert isinstance(rng, tuple)
            assert len(rng) == 2
            assert rng[0] < rng[1]


# --- compute_risk_level ---

class TestComputeRiskLevel:
    """Tests para clasificación de riesgo físico."""

    def test_within_range_is_low(self) -> None:
        assert compute_risk_level("temperature", 25.0) == "LOW"

    def test_slightly_outside_is_medium(self) -> None:
        # temperature range: (15, 35), margin = 0.1 * 20 = 2.0
        # 36.0 is outside [15, 35] but within [13, 37] → MEDIUM
        assert compute_risk_level("temperature", 36.0) == "MEDIUM"

    def test_far_outside_is_high(self) -> None:
        # temperature range: (15, 35), margin = 2.0
        # 40.0 > 37.0 → HIGH
        assert compute_risk_level("temperature", 40.0) == "HIGH"

    def test_below_range_medium(self) -> None:
        assert compute_risk_level("temperature", 14.0) == "MEDIUM"

    def test_far_below_range_high(self) -> None:
        assert compute_risk_level("temperature", 10.0) == "HIGH"

    def test_unknown_sensor_type(self) -> None:
        assert compute_risk_level("unknown", 100.0) == "NONE"

    def test_exact_boundary_is_low(self) -> None:
        assert compute_risk_level("temperature", 15.0) == "LOW"
        assert compute_risk_level("temperature", 35.0) == "LOW"


# --- compute_severity ---

class TestComputeSeverity:
    """Tests para combinación anomalía + riesgo → severidad."""

    def test_out_of_range_is_critical(self) -> None:
        assert compute_severity(
            is_anomaly=False, risk_level="LOW", out_of_physical_range=True
        ) == "critical"

    def test_anomaly_plus_high_risk_is_critical(self) -> None:
        assert compute_severity(
            is_anomaly=True, risk_level="HIGH", out_of_physical_range=False
        ) == "critical"

    def test_anomaly_alone_is_warning(self) -> None:
        assert compute_severity(
            is_anomaly=True, risk_level="LOW", out_of_physical_range=False
        ) == "warning"

    def test_high_risk_alone_is_warning(self) -> None:
        assert compute_severity(
            is_anomaly=False, risk_level="HIGH", out_of_physical_range=False
        ) == "warning"

    def test_normal_is_info(self) -> None:
        assert compute_severity(
            is_anomaly=False, risk_level="LOW", out_of_physical_range=False
        ) == "info"

    def test_none_risk_no_anomaly_is_info(self) -> None:
        assert compute_severity(
            is_anomaly=False, risk_level="NONE", out_of_physical_range=False
        ) == "info"

    def test_out_of_range_overrides_all(self) -> None:
        """out_of_physical_range siempre es critical, sin importar otros."""
        assert compute_severity(
            is_anomaly=False, risk_level="NONE", out_of_physical_range=True
        ) == "critical"


# --- is_out_of_range ---

class TestIsOutOfRange:
    """Tests para verificación de rango."""

    def test_within_range(self) -> None:
        assert is_out_of_range(25.0, (20.0, 30.0)) is False

    def test_below_range(self) -> None:
        assert is_out_of_range(15.0, (20.0, 30.0)) is True

    def test_above_range(self) -> None:
        assert is_out_of_range(35.0, (20.0, 30.0)) is True

    def test_none_range(self) -> None:
        assert is_out_of_range(25.0, None) is False

    def test_exact_boundary(self) -> None:
        assert is_out_of_range(20.0, (20.0, 30.0)) is False
        assert is_out_of_range(30.0, (20.0, 30.0)) is False


# --- build_recommended_action ---

class TestBuildRecommendedAction:
    """Tests para generación de acción recomendada."""

    def test_info_low_risk(self) -> None:
        action = build_recommended_action(
            severity="info", risk_level="LOW", location="Lab"
        )
        assert "No se requiere acción" in action

    def test_info_medium_risk(self) -> None:
        action = build_recommended_action(
            severity="info", risk_level="MEDIUM", location="Lab"
        )
        assert "límites operativos" in action
        assert "Lab" in action

    def test_critical(self) -> None:
        action = build_recommended_action(
            severity="critical", risk_level="HIGH", location="Sala A"
        )
        assert "crítica" in action
        assert "Sala A" in action

    def test_warning_high_risk(self) -> None:
        action = build_recommended_action(
            severity="warning", risk_level="HIGH", location="Planta"
        )
        assert "Riesgo elevado" in action
        assert "Planta" in action

    def test_warning_low_risk(self) -> None:
        action = build_recommended_action(
            severity="warning", risk_level="LOW", location="Bodega"
        )
        assert "Comportamiento inusual" in action
        assert "Bodega" in action


# --- classify_severity (integración de dominio) ---

class TestClassifySeverity:
    """Tests para clasificación completa."""

    def test_normal_temperature(self) -> None:
        result = classify_severity(
            sensor_type="temperature",
            location="Lab",
            predicted_value=25.0,
            anomaly=False,
        )
        assert isinstance(result, SeverityResult)
        assert result.severity == "info"
        assert result.action_required is False

    def test_extreme_temperature_is_critical(self) -> None:
        result = classify_severity(
            sensor_type="temperature",
            location="Lab",
            predicted_value=50.0,
            anomaly=False,
        )
        assert result.severity == "critical"
        assert result.action_required is True

    def test_anomaly_with_high_risk(self) -> None:
        result = classify_severity(
            sensor_type="temperature",
            location="Lab",
            predicted_value=50.0,
            anomaly=True,
        )
        assert result.severity == "critical"

    def test_anomaly_within_range(self) -> None:
        result = classify_severity(
            sensor_type="temperature",
            location="Lab",
            predicted_value=25.0,
            anomaly=True,
        )
        assert result.severity == "warning"

    def test_user_range_overrides_default(self) -> None:
        """Rango del usuario tiene prioridad sobre DEFAULT_RANGES."""
        result = classify_severity(
            sensor_type="temperature",
            location="Lab",
            predicted_value=25.0,
            anomaly=False,
            user_defined_range=(10.0, 20.0),  # 25 está fuera
        )
        assert result.severity == "critical"

    def test_unknown_sensor_no_range(self) -> None:
        result = classify_severity(
            sensor_type="custom_sensor",
            location="Lab",
            predicted_value=999.0,
            anomaly=False,
        )
        assert result.risk_level == "NONE"
        assert result.severity == "info"

    def test_unknown_sensor_with_anomaly(self) -> None:
        result = classify_severity(
            sensor_type="custom_sensor",
            location="Lab",
            predicted_value=999.0,
            anomaly=True,
        )
        assert result.severity == "warning"
