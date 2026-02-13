"""Tests para compute_temporal_features y TemporalFeatures value object.

Cubre:
- Señales vacías y con 1 punto
- Velocidad constante (señal lineal)
- Aceleración constante (señal cuadrática)
- Muestreo uniforme vs irregular (jitter)
- Señal constante (velocidad = 0)
- Señal con spike abrupto
- Serialización to_dict y to_feature_vector
- Edge cases: timestamps duplicados, Δt muy pequeño
- Integración con TimeSeries.temporal_features
- Integración con SensorWindow.temporal_features
"""

from __future__ import annotations

import math

import pytest

from iot_machine_learning.domain.entities.temporal_features import TemporalFeatures
from iot_machine_learning.domain.validators.temporal_features import (
    compute_temporal_features,
    _compute_dt_values,
    _compute_velocities,
    _compute_accelerations,
    _compute_jitter,
    _stats,
)


# ── TemporalFeatures value object ──────────────────────────────────────


class TestTemporalFeaturesObject:
    """Tests para el value object TemporalFeatures."""

    def test_empty_factory(self):
        tf = TemporalFeatures.empty()
        assert tf.n_points == 0
        assert tf.velocities == []
        assert tf.accelerations == []
        assert tf.has_velocity is False
        assert tf.has_acceleration is False

    def test_has_velocity(self):
        tf = TemporalFeatures(velocities=[1.0, 2.0], n_points=3)
        assert tf.has_velocity is True

    def test_has_acceleration(self):
        tf = TemporalFeatures(accelerations=[0.5], n_points=3)
        assert tf.has_acceleration is True

    def test_is_uniform_sampling_low_jitter(self):
        tf = TemporalFeatures(jitter=0.05)
        assert tf.is_uniform_sampling is True

    def test_is_uniform_sampling_high_jitter(self):
        tf = TemporalFeatures(jitter=0.3)
        assert tf.is_uniform_sampling is False

    def test_to_feature_vector_length(self):
        tf = TemporalFeatures(
            mean_velocity=1.0, std_velocity=0.5, max_abs_velocity=2.0,
            mean_acceleration=0.1, std_acceleration=0.05, max_abs_acceleration=0.2,
            jitter=0.01, last_velocity=1.5, last_acceleration=0.08,
        )
        vec = tf.to_feature_vector()
        assert len(vec) == 9
        assert vec[0] == 1.0  # mean_velocity
        assert vec[6] == 0.01  # jitter

    def test_to_dict_keys(self):
        tf = TemporalFeatures(n_points=10, velocities=[1.0] * 9)
        d = tf.to_dict()
        expected_keys = {
            "mean_velocity", "std_velocity", "max_abs_velocity",
            "mean_acceleration", "std_acceleration", "max_abs_acceleration",
            "jitter", "mean_dt", "last_velocity", "last_acceleration",
            "n_points", "n_velocities", "n_accelerations",
        }
        assert set(d.keys()) == expected_keys
        assert d["n_points"] == 10
        assert d["n_velocities"] == 9

    def test_frozen(self):
        tf = TemporalFeatures()
        with pytest.raises(AttributeError):
            tf.n_points = 5  # type: ignore[misc]


# ── Helper functions ───────────────────────────────────────────────────


class TestHelperFunctions:
    """Tests para funciones internas de cómputo."""

    def test_compute_dt_values_uniform(self):
        ts = [0.0, 1.0, 2.0, 3.0]
        dts = _compute_dt_values(ts)
        assert len(dts) == 3
        assert all(abs(dt - 1.0) < 1e-9 for dt in dts)

    def test_compute_dt_values_irregular(self):
        ts = [0.0, 1.0, 5.0, 6.0]
        dts = _compute_dt_values(ts)
        assert abs(dts[0] - 1.0) < 1e-9
        assert abs(dts[1] - 4.0) < 1e-9
        assert abs(dts[2] - 1.0) < 1e-9

    def test_compute_dt_values_duplicate_timestamps(self):
        ts = [0.0, 0.0, 1.0]
        dts = _compute_dt_values(ts)
        assert dts[0] < 1e-6  # epsilon, not zero
        assert abs(dts[1] - 1.0) < 1e-9

    def test_compute_velocities_linear(self):
        values = [0.0, 2.0, 4.0, 6.0]
        dts = [1.0, 1.0, 1.0]
        vels = _compute_velocities(values, dts)
        assert len(vels) == 3
        assert all(abs(v - 2.0) < 1e-9 for v in vels)

    def test_compute_velocities_with_variable_dt(self):
        values = [0.0, 10.0, 30.0]
        dts = [1.0, 2.0]
        vels = _compute_velocities(values, dts)
        assert abs(vels[0] - 10.0) < 1e-9  # 10/1
        assert abs(vels[1] - 10.0) < 1e-9  # 20/2

    def test_compute_accelerations_constant_velocity(self):
        vels = [5.0, 5.0, 5.0]
        dts = [1.0, 1.0, 1.0]
        accs = _compute_accelerations(vels, dts)
        assert len(accs) == 2
        assert all(abs(a) < 1e-9 for a in accs)

    def test_compute_accelerations_linear_velocity(self):
        # v = [0, 2, 4] → a = 2.0 everywhere
        vels = [0.0, 2.0, 4.0]
        dts = [1.0, 1.0, 1.0]
        accs = _compute_accelerations(vels, dts)
        assert len(accs) == 2
        assert all(abs(a - 2.0) < 1e-9 for a in accs)

    def test_compute_accelerations_insufficient(self):
        assert _compute_accelerations([5.0], [1.0]) == []
        assert _compute_accelerations([], []) == []

    def test_compute_jitter_uniform(self):
        dts = [1.0, 1.0, 1.0, 1.0]
        assert _compute_jitter(dts) == 0.0

    def test_compute_jitter_variable(self):
        dts = [1.0, 2.0, 1.0, 2.0]
        j = _compute_jitter(dts)
        assert j > 0.0
        assert j < 1.0

    def test_compute_jitter_single(self):
        assert _compute_jitter([1.0]) == 0.0

    def test_stats_empty(self):
        assert _stats([]) == (0.0, 0.0, 0.0)

    def test_stats_single(self):
        mean_abs, std, max_abs = _stats([3.0])
        assert abs(mean_abs - 3.0) < 1e-9
        assert abs(max_abs - 3.0) < 1e-9

    def test_stats_mixed(self):
        mean_abs, std, max_abs = _stats([-2.0, 4.0])
        assert abs(mean_abs - 3.0) < 1e-9  # (2+4)/2
        assert abs(max_abs - 4.0) < 1e-9


# ── compute_temporal_features integration ──────────────────────────────


class TestComputeTemporalFeatures:
    """Tests para la función principal compute_temporal_features."""

    def test_empty_series(self):
        tf = compute_temporal_features([], [])
        assert tf.n_points == 0
        assert tf == TemporalFeatures.empty()

    def test_single_point(self):
        tf = compute_temporal_features([42.0], [100.0])
        assert tf.n_points == 1
        assert tf.has_velocity is False
        assert tf.has_acceleration is False

    def test_two_points(self):
        tf = compute_temporal_features([10.0, 20.0], [0.0, 2.0])
        assert tf.n_points == 2
        assert tf.has_velocity is True
        assert tf.has_acceleration is False
        assert abs(tf.velocities[0] - 5.0) < 1e-9  # 10/2
        assert abs(tf.last_velocity - 5.0) < 1e-9
        assert abs(tf.mean_dt - 2.0) < 1e-9

    def test_linear_signal_constant_velocity(self):
        # v(t) = 3t, dt=1 → velocity = 3.0 everywhere
        values = [0.0, 3.0, 6.0, 9.0, 12.0]
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        tf = compute_temporal_features(values, timestamps)

        assert tf.n_points == 5
        assert len(tf.velocities) == 4
        assert all(abs(v - 3.0) < 1e-9 for v in tf.velocities)
        assert abs(tf.mean_velocity - 3.0) < 1e-9
        assert abs(tf.std_velocity) < 1e-9
        assert abs(tf.max_abs_velocity - 3.0) < 1e-9

        # Acceleration should be ~0
        assert len(tf.accelerations) == 3
        assert all(abs(a) < 1e-9 for a in tf.accelerations)
        assert abs(tf.mean_acceleration) < 1e-9

    def test_quadratic_signal_constant_acceleration(self):
        # v(t) = t², dt=1 → velocity = 2t-1, acceleration = 2
        values = [0.0, 1.0, 4.0, 9.0, 16.0]
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        tf = compute_temporal_features(values, timestamps)

        # velocities: [1, 3, 5, 7]
        assert len(tf.velocities) == 4
        assert abs(tf.velocities[0] - 1.0) < 1e-9
        assert abs(tf.velocities[1] - 3.0) < 1e-9
        assert abs(tf.velocities[2] - 5.0) < 1e-9
        assert abs(tf.velocities[3] - 7.0) < 1e-9

        # accelerations: [2, 2, 2]
        assert len(tf.accelerations) == 3
        assert all(abs(a - 2.0) < 1e-9 for a in tf.accelerations)
        assert abs(tf.last_acceleration - 2.0) < 1e-9

    def test_constant_signal_zero_velocity(self):
        values = [25.0] * 10
        timestamps = [float(i) for i in range(10)]
        tf = compute_temporal_features(values, timestamps)

        assert all(abs(v) < 1e-9 for v in tf.velocities)
        assert abs(tf.mean_velocity) < 1e-9
        assert abs(tf.max_abs_velocity) < 1e-9

    def test_spike_signal_high_velocity(self):
        # Normal → spike → normal
        values = [20.0, 20.0, 20.0, 100.0, 20.0, 20.0]
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        tf = compute_temporal_features(values, timestamps)

        # velocity at spike: (100-20)/1 = 80
        assert abs(tf.velocities[2] - 80.0) < 1e-9
        # velocity after spike: (20-100)/1 = -80
        assert abs(tf.velocities[3] - (-80.0)) < 1e-9
        assert abs(tf.max_abs_velocity - 80.0) < 1e-9

    def test_irregular_sampling_jitter(self):
        values = [0.0, 1.0, 2.0, 3.0]
        timestamps = [0.0, 1.0, 10.0, 11.0]  # gap between t=1 and t=10
        tf = compute_temporal_features(values, timestamps)

        assert tf.jitter > 0.5  # very irregular
        assert tf.is_uniform_sampling is False

    def test_uniform_sampling_zero_jitter(self):
        values = [0.0, 1.0, 2.0, 3.0, 4.0]
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        tf = compute_temporal_features(values, timestamps)

        assert abs(tf.jitter) < 1e-9
        assert tf.is_uniform_sampling is True

    def test_velocity_with_variable_dt(self):
        # Same Δv but different Δt → different velocities
        values = [0.0, 10.0, 20.0]
        timestamps = [0.0, 1.0, 11.0]  # dt=1, dt=10
        tf = compute_temporal_features(values, timestamps)

        assert abs(tf.velocities[0] - 10.0) < 1e-9   # 10/1
        assert abs(tf.velocities[1] - 1.0) < 1e-9     # 10/10

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="misma longitud"):
            compute_temporal_features([1.0, 2.0], [1.0])

    def test_mean_dt(self):
        tf = compute_temporal_features(
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 2.0, 4.0, 6.0],
        )
        assert abs(tf.mean_dt - 2.0) < 1e-9

    def test_last_velocity_and_acceleration(self):
        values = [0.0, 1.0, 4.0, 9.0]
        timestamps = [0.0, 1.0, 2.0, 3.0]
        tf = compute_temporal_features(values, timestamps)

        # last velocity = (9-4)/1 = 5
        assert abs(tf.last_velocity - 5.0) < 1e-9
        # velocities = [1, 3, 5], accelerations = [2, 2]
        assert abs(tf.last_acceleration - 2.0) < 1e-9

    def test_negative_velocity(self):
        values = [10.0, 8.0, 6.0]
        timestamps = [0.0, 1.0, 2.0]
        tf = compute_temporal_features(values, timestamps)

        assert all(abs(v - (-2.0)) < 1e-9 for v in tf.velocities)
        assert abs(tf.mean_velocity - 2.0) < 1e-9  # mean of |v|

    def test_to_feature_vector_consistency(self):
        values = [0.0, 1.0, 4.0, 9.0, 16.0]
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        tf = compute_temporal_features(values, timestamps)
        vec = tf.to_feature_vector()

        assert vec[0] == tf.mean_velocity
        assert vec[1] == tf.std_velocity
        assert vec[2] == tf.max_abs_velocity
        assert vec[3] == tf.mean_acceleration
        assert vec[7] == tf.last_velocity
        assert vec[8] == tf.last_acceleration


# ── TimeSeries integration ─────────────────────────────────────────────


class TestTimeSeriesTemporalFeatures:
    """Tests para TimeSeries.temporal_features property."""

    def test_temporal_features_from_time_series(self):
        from iot_machine_learning.domain.entities.time_series import TimeSeries

        ts = TimeSeries.from_values(
            values=[0.0, 3.0, 6.0, 9.0],
            timestamps=[0.0, 1.0, 2.0, 3.0],
        )
        tf = ts.temporal_features

        assert tf.n_points == 4
        assert tf.has_velocity is True
        assert all(abs(v - 3.0) < 1e-9 for v in tf.velocities)

    def test_temporal_features_empty_series(self):
        from iot_machine_learning.domain.entities.time_series import TimeSeries

        ts = TimeSeries(series_id="empty", points=[])
        tf = ts.temporal_features
        assert tf.n_points == 0
        assert tf.has_velocity is False


# ── SensorWindow integration ──────────────────────────────────────────


class TestSensorWindowTemporalFeatures:
    """Tests para SensorWindow.temporal_features property."""

    def test_temporal_features_from_sensor_window(self):
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading,
            SensorWindow,
        )

        readings = [
            SensorReading(sensor_id=1, value=float(i * 5), timestamp=float(i))
            for i in range(5)
        ]
        sw = SensorWindow(sensor_id=1, readings=readings)
        tf = sw.temporal_features

        assert tf.n_points == 5
        assert tf.has_velocity is True
        # velocity = 5/1 = 5.0 for all pairs
        assert all(abs(v - 5.0) < 1e-9 for v in tf.velocities)

    def test_temporal_features_empty_window(self):
        from iot_machine_learning.domain.entities.sensor_reading import SensorWindow

        sw = SensorWindow(sensor_id=1, readings=[])
        tf = sw.temporal_features
        assert tf.n_points == 0

    def test_temporal_features_matches_time_series(self):
        """SensorWindow.temporal_features should match TimeSeries.temporal_features."""
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading,
            SensorWindow,
        )

        readings = [
            SensorReading(sensor_id=1, value=float(i ** 2), timestamp=float(i))
            for i in range(6)
        ]
        sw = SensorWindow(sensor_id=1, readings=readings)
        ts = sw.to_time_series()

        tf_sw = sw.temporal_features
        tf_ts = ts.temporal_features

        assert tf_sw.n_points == tf_ts.n_points
        assert len(tf_sw.velocities) == len(tf_ts.velocities)
        for v1, v2 in zip(tf_sw.velocities, tf_ts.velocities):
            assert abs(v1 - v2) < 1e-9
        assert abs(tf_sw.jitter - tf_ts.jitter) < 1e-9
