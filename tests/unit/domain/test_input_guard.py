"""Tests para domain/validators/input_guard.py — guards de entrada."""

from __future__ import annotations

import math
import time

import pytest

from iot_machine_learning.domain.validators.input_guard import (
    InputGuardError,
    guard_finite_value,
    guard_no_future_timestamps,
    guard_sensor_id,
    guard_series_id,
    guard_window_size,
)


class TestGuardSensorId:

    def test_valid_int(self) -> None:
        assert guard_sensor_id(42) == 42

    def test_valid_string_int(self) -> None:
        assert guard_sensor_id("42") == 42

    def test_none_raises(self) -> None:
        with pytest.raises(InputGuardError, match="requerido"):
            guard_sensor_id(None)

    def test_zero_raises(self) -> None:
        with pytest.raises(InputGuardError, match="positivo"):
            guard_sensor_id(0)

    def test_negative_raises(self) -> None:
        with pytest.raises(InputGuardError, match="positivo"):
            guard_sensor_id(-1)

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(InputGuardError, match="entero"):
            guard_sensor_id("abc")


class TestGuardSeriesId:

    def test_valid_string(self) -> None:
        assert guard_series_id("sensor-42") == "sensor-42"

    def test_int_converted(self) -> None:
        assert guard_series_id(42) == "42"

    def test_none_raises(self) -> None:
        with pytest.raises(InputGuardError, match="requerido"):
            guard_series_id(None)

    def test_empty_raises(self) -> None:
        with pytest.raises(InputGuardError, match="vacío"):
            guard_series_id("")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(InputGuardError, match="vacío"):
            guard_series_id("   ")


class TestGuardWindowSize:

    def test_valid(self) -> None:
        assert guard_window_size(500) == 500

    def test_min_boundary(self) -> None:
        assert guard_window_size(1) == 1

    def test_below_min_raises(self) -> None:
        with pytest.raises(InputGuardError, match=">= 1"):
            guard_window_size(0)

    def test_above_max_raises(self) -> None:
        with pytest.raises(InputGuardError, match="<= 10000"):
            guard_window_size(20_000)

    def test_custom_bounds(self) -> None:
        assert guard_window_size(50, min_size=10, max_size=100) == 50

    def test_non_int_raises(self) -> None:
        with pytest.raises(InputGuardError, match="entero"):
            guard_window_size(3.14)  # type: ignore[arg-type]


class TestGuardNoFutureTimestamps:

    def test_past_timestamps_pass(self) -> None:
        now = time.time()
        guard_no_future_timestamps([now - 100, now - 50, now - 10])

    def test_near_future_within_tolerance(self) -> None:
        now = time.time()
        guard_no_future_timestamps([now + 60], tolerance_seconds=300.0)

    def test_far_future_raises(self) -> None:
        now = time.time()
        with pytest.raises(InputGuardError, match="futuro"):
            guard_no_future_timestamps([now + 1000], tolerance_seconds=300.0)

    def test_empty_list_passes(self) -> None:
        guard_no_future_timestamps([])


class TestGuardFiniteValue:

    def test_valid_float(self) -> None:
        assert guard_finite_value(3.14) == 3.14

    def test_valid_int(self) -> None:
        assert guard_finite_value(42) == 42.0

    def test_none_raises(self) -> None:
        with pytest.raises(InputGuardError, match="requerido"):
            guard_finite_value(None)

    def test_nan_raises(self) -> None:
        with pytest.raises(InputGuardError, match="finito"):
            guard_finite_value(float("nan"))

    def test_inf_raises(self) -> None:
        with pytest.raises(InputGuardError, match="finito"):
            guard_finite_value(float("inf"))

    def test_string_raises(self) -> None:
        with pytest.raises(InputGuardError, match="numérico"):
            guard_finite_value("abc")

    def test_custom_name_in_error(self) -> None:
        with pytest.raises(InputGuardError, match="predicted_value"):
            guard_finite_value(None, name="predicted_value")
