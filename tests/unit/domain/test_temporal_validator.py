"""Tests para domain/validators/temporal.py — validaciones temporales."""

from __future__ import annotations

import math
import pytest

from iot_machine_learning.domain.validators.temporal import (
    TemporalDiagnostic,
    TemporalValidationError,
    diagnose_temporal_quality,
    sort_and_deduplicate,
    validate_timestamps,
)


class TestValidateTimestamps:
    """Tests para validate_timestamps."""

    def test_empty_list_raises(self) -> None:
        with pytest.raises(TemporalValidationError, match="vacía"):
            validate_timestamps([])

    def test_non_numeric_raises(self) -> None:
        with pytest.raises(TemporalValidationError, match="no es numérico"):
            validate_timestamps([1.0, "bad", 3.0])  # type: ignore[list-item]

    def test_nan_raises(self) -> None:
        with pytest.raises(TemporalValidationError, match="no es finito"):
            validate_timestamps([1.0, float("nan"), 3.0])

    def test_inf_raises(self) -> None:
        with pytest.raises(TemporalValidationError, match="no es finito"):
            validate_timestamps([1.0, float("inf")])

    def test_negative_timestamp_raises_when_required(self) -> None:
        with pytest.raises(TemporalValidationError, match="no es positivo"):
            validate_timestamps([-1.0, 2.0], require_positive=True)

    def test_negative_timestamp_ok_when_not_required(self) -> None:
        validate_timestamps([-1.0, 2.0], require_positive=False)

    def test_zero_timestamp_raises_when_positive_required(self) -> None:
        with pytest.raises(TemporalValidationError, match="no es positivo"):
            validate_timestamps([0.0, 1.0], require_positive=True)

    def test_valid_timestamps_pass(self) -> None:
        validate_timestamps([1.0, 2.0, 3.0])

    def test_non_monotonic_raises_when_required(self) -> None:
        with pytest.raises(TemporalValidationError, match="no monótonos"):
            validate_timestamps([1.0, 3.0, 2.0], require_monotonic=True)

    def test_duplicate_raises_when_monotonic_required(self) -> None:
        with pytest.raises(TemporalValidationError, match="no monótonos"):
            validate_timestamps([1.0, 2.0, 2.0], require_monotonic=True)

    def test_monotonic_passes(self) -> None:
        validate_timestamps([1.0, 2.0, 3.0], require_monotonic=True)


class TestDiagnoseTemporalQuality:
    """Tests para diagnose_temporal_quality."""

    def test_single_point(self) -> None:
        diag = diagnose_temporal_quality([1.0])
        assert diag.is_monotonic is True
        assert diag.is_clean is True
        assert diag.median_dt == 0.0

    def test_two_points(self) -> None:
        diag = diagnose_temporal_quality([1.0, 2.0])
        assert diag.is_monotonic is True
        assert diag.median_dt == 1.0
        assert diag.n_gaps == 0

    def test_out_of_order_detected(self) -> None:
        diag = diagnose_temporal_quality([1.0, 3.0, 2.0, 4.0])
        assert diag.is_monotonic is False
        assert diag.n_out_of_order == 1
        assert diag.is_clean is False

    def test_duplicates_detected(self) -> None:
        diag = diagnose_temporal_quality([1.0, 2.0, 2.0, 3.0])
        assert diag.n_duplicates == 1
        assert diag.is_monotonic is False
        assert diag.is_clean is False

    def test_gap_detected(self) -> None:
        # Regular interval of 1.0, then a gap of 20.0
        ts = [1.0, 2.0, 3.0, 4.0, 5.0, 25.0]
        diag = diagnose_temporal_quality(ts, max_gap_factor=5.0)
        assert diag.n_gaps == 1
        assert 5 in diag.gap_indices  # index of point after gap

    def test_no_gap_within_factor(self) -> None:
        ts = [1.0, 2.0, 3.0, 4.0, 5.0, 8.0]
        diag = diagnose_temporal_quality(ts, max_gap_factor=5.0)
        assert diag.n_gaps == 0

    def test_clean_series(self) -> None:
        ts = [float(i) for i in range(1, 101)]
        diag = diagnose_temporal_quality(ts)
        assert diag.is_clean is True
        assert diag.median_dt == 1.0
        assert diag.min_dt == 1.0
        assert diag.max_dt == 1.0

    def test_all_same_timestamps(self) -> None:
        diag = diagnose_temporal_quality([5.0, 5.0, 5.0])
        assert diag.n_duplicates == 2
        assert diag.is_monotonic is False


class TestSortAndDeduplicate:
    """Tests para sort_and_deduplicate."""

    def test_already_sorted_no_dupes(self) -> None:
        ts, vals = sort_and_deduplicate([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        assert ts == [1.0, 2.0, 3.0]
        assert vals == [10.0, 20.0, 30.0]

    def test_out_of_order_sorted(self) -> None:
        ts, vals = sort_and_deduplicate([3.0, 1.0, 2.0], [30.0, 10.0, 20.0])
        assert ts == [1.0, 2.0, 3.0]
        assert vals == [10.0, 20.0, 30.0]

    def test_duplicates_keep_last(self) -> None:
        ts, vals = sort_and_deduplicate(
            [1.0, 2.0, 2.0, 3.0], [10.0, 20.0, 25.0, 30.0]
        )
        assert ts == [1.0, 2.0, 3.0]
        assert vals == [10.0, 25.0, 30.0]  # 25.0 overwrites 20.0

    def test_empty_input(self) -> None:
        ts, vals = sort_and_deduplicate([], [])
        assert ts == []
        assert vals == []

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="misma longitud"):
            sort_and_deduplicate([1.0, 2.0], [10.0])

    def test_single_element(self) -> None:
        ts, vals = sort_and_deduplicate([5.0], [50.0])
        assert ts == [5.0]
        assert vals == [50.0]


class TestSensorReadingTimestampValidation:
    """Tests para la validación de timestamp en SensorReading."""

    def test_infinite_timestamp_raises(self) -> None:
        from iot_machine_learning.domain.entities.sensor_reading import SensorReading

        with pytest.raises(ValueError, match="timestamp debe ser finito"):
            SensorReading(sensor_id=1, value=25.0, timestamp=float("inf"))

    def test_nan_timestamp_raises(self) -> None:
        from iot_machine_learning.domain.entities.sensor_reading import SensorReading

        with pytest.raises(ValueError, match="timestamp debe ser finito"):
            SensorReading(sensor_id=1, value=25.0, timestamp=float("nan"))

    def test_valid_timestamp_passes(self) -> None:
        from iot_machine_learning.domain.entities.sensor_reading import SensorReading

        r = SensorReading(sensor_id=1, value=25.0, timestamp=1000.0)
        assert r.timestamp == 1000.0

    def test_zero_timestamp_allowed(self) -> None:
        from iot_machine_learning.domain.entities.sensor_reading import SensorReading

        r = SensorReading(sensor_id=1, value=25.0, timestamp=0.0)
        assert r.timestamp == 0.0


class TestSensorWindowTemporalDiagnostic:
    """Tests para SensorWindow.temporal_diagnostic."""

    def test_clean_window(self) -> None:
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading,
            SensorWindow,
        )

        readings = [
            SensorReading(sensor_id=1, value=float(i), timestamp=float(i))
            for i in range(1, 11)
        ]
        window = SensorWindow(sensor_id=1, readings=readings)
        diag = window.temporal_diagnostic
        assert diag.is_clean is True

    def test_empty_window(self) -> None:
        from iot_machine_learning.domain.entities.sensor_reading import SensorWindow

        window = SensorWindow(sensor_id=1, readings=[])
        diag = window.temporal_diagnostic
        assert diag.is_clean is True
