"""Tests para domain/services/threshold_evaluator.py.

Verifica reglas de negocio PURAS — sin I/O, sin mocks de BD.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.services.threshold_evaluator import (
    ThresholdDefinition,
    ThresholdViolation,
    build_violation,
    is_threshold_violated,
    is_within_warning_range,
)


# --- Fixtures ---

def _threshold(
    condition_type: str = "greater_than",
    value_min: float | None = 30.0,
    value_max: float | None = None,
    severity: str = "warning",
    name: str = "test_threshold",
) -> ThresholdDefinition:
    return ThresholdDefinition(
        threshold_id=1,
        name=name,
        condition_type=condition_type,
        value_min=value_min,
        value_max=value_max,
        severity=severity,
    )


# --- is_threshold_violated ---

class TestIsThresholdViolated:
    """Tests para evaluación de violación de umbral."""

    def test_greater_than_violated(self) -> None:
        thr = _threshold(condition_type="greater_than", value_min=30.0)
        assert is_threshold_violated(31.0, thr) is True

    def test_greater_than_not_violated(self) -> None:
        thr = _threshold(condition_type="greater_than", value_min=30.0)
        assert is_threshold_violated(29.0, thr) is False

    def test_greater_than_exact_boundary(self) -> None:
        thr = _threshold(condition_type="greater_than", value_min=30.0)
        assert is_threshold_violated(30.0, thr) is False

    def test_less_than_violated(self) -> None:
        thr = _threshold(condition_type="less_than", value_min=10.0)
        assert is_threshold_violated(9.0, thr) is True

    def test_less_than_not_violated(self) -> None:
        thr = _threshold(condition_type="less_than", value_min=10.0)
        assert is_threshold_violated(11.0, thr) is False

    def test_out_of_range_below(self) -> None:
        thr = _threshold(condition_type="out_of_range", value_min=10.0, value_max=30.0)
        assert is_threshold_violated(5.0, thr) is True

    def test_out_of_range_above(self) -> None:
        thr = _threshold(condition_type="out_of_range", value_min=10.0, value_max=30.0)
        assert is_threshold_violated(35.0, thr) is True

    def test_out_of_range_within(self) -> None:
        thr = _threshold(condition_type="out_of_range", value_min=10.0, value_max=30.0)
        assert is_threshold_violated(20.0, thr) is False

    def test_equal_to_violated(self) -> None:
        thr = _threshold(condition_type="equal_to", value_min=42.0)
        assert is_threshold_violated(42.0, thr) is True

    def test_equal_to_not_violated(self) -> None:
        thr = _threshold(condition_type="equal_to", value_min=42.0)
        assert is_threshold_violated(43.0, thr) is False

    def test_unknown_condition_not_violated(self) -> None:
        thr = _threshold(condition_type="unknown_type", value_min=10.0)
        assert is_threshold_violated(10.0, thr) is False

    def test_none_min_greater_than(self) -> None:
        thr = _threshold(condition_type="greater_than", value_min=None)
        assert is_threshold_violated(100.0, thr) is False


# --- is_within_warning_range ---

class TestIsWithinWarningRange:
    """Tests para verificación de rango WARNING."""

    def test_within_range(self) -> None:
        assert is_within_warning_range(25.0, 20.0, 30.0) is True

    def test_below_range(self) -> None:
        assert is_within_warning_range(15.0, 20.0, 30.0) is False

    def test_above_range(self) -> None:
        assert is_within_warning_range(35.0, 20.0, 30.0) is False

    def test_no_range_defined(self) -> None:
        assert is_within_warning_range(25.0, None, None) is False

    def test_only_min_defined_within(self) -> None:
        assert is_within_warning_range(25.0, 20.0, None) is True

    def test_only_min_defined_below(self) -> None:
        assert is_within_warning_range(15.0, 20.0, None) is False

    def test_only_max_defined_within(self) -> None:
        assert is_within_warning_range(25.0, None, 30.0) is True

    def test_only_max_defined_above(self) -> None:
        assert is_within_warning_range(35.0, None, 30.0) is False

    def test_exact_boundary_min(self) -> None:
        assert is_within_warning_range(20.0, 20.0, 30.0) is True

    def test_exact_boundary_max(self) -> None:
        assert is_within_warning_range(30.0, 20.0, 30.0) is True


# --- build_violation ---

class TestBuildViolation:
    """Tests para construcción de ThresholdViolation."""

    def test_critical_severity(self) -> None:
        thr = _threshold(severity="critical", name="Temp Alta")
        v = build_violation(35.0, thr)
        assert isinstance(v, ThresholdViolation)
        assert v.event_type == "critical"
        assert "Temp Alta" in v.title
        assert v.threshold_id == 1

    def test_warning_severity(self) -> None:
        thr = _threshold(severity="warning")
        v = build_violation(35.0, thr)
        assert v.event_type == "warning"

    def test_info_severity(self) -> None:
        thr = _threshold(severity="info")
        v = build_violation(35.0, thr)
        assert v.event_type == "notice"

    def test_payload_contains_predicted_value(self) -> None:
        thr = _threshold(value_min=30.0, value_max=40.0)
        v = build_violation(45.0, thr)
        assert v.payload["predicted_value"] == 45.0
        assert v.payload["threshold_value_min"] == 30.0
        assert v.payload["threshold_value_max"] == 40.0

    def test_message_contains_ids(self) -> None:
        thr = _threshold()
        v = build_violation(35.0, thr)
        assert "predicted_value=35.0" in v.message
        assert "threshold_id=1" in v.message
