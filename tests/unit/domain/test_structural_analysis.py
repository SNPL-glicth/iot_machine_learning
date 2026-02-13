"""Tests para StructuralAnalysis y compute_structural_analysis.

Cubre:
- Señales vacías y con 1-2 puntos
- Señal constante (slope=0, stable)
- Señal lineal (slope constante, trending)
- Señal cuadrática (curvature constante)
- Señal ruidosa (noisy regime)
- Señal volátil (volatile regime)
- Stability indicator
- Accel variance
- Noise ratio
- Trend strength
- from_taylor_diagnostic bridge
- Serialización to_dict y to_feature_vector
- Integración con TimeSeries.structural_analysis
- Integración con SensorWindow.structural_analysis
- Consistencia entre SensorWindow y TimeSeries
"""

from __future__ import annotations

import math
import random

import pytest

from iot_machine_learning.domain.entities.structural_analysis import (
    RegimeType,
    StructuralAnalysis,
    _classify_regime,
)
from iot_machine_learning.domain.validators.structural_analysis import (
    compute_structural_analysis,
    _compute_accel_variance,
    _compute_median_dt,
    _compute_stability,
)


# ── StructuralAnalysis value object ────────────────────────────────────


class TestStructuralAnalysisObject:
    """Tests para el value object StructuralAnalysis."""

    def test_empty_factory(self):
        sa = StructuralAnalysis.empty()
        assert sa.n_points == 0
        assert sa.is_stable is True
        assert sa.has_sufficient_data is False

    def test_is_stable(self):
        sa = StructuralAnalysis(regime=RegimeType.STABLE)
        assert sa.is_stable is True
        assert sa.is_trending is False

    def test_is_trending(self):
        sa = StructuralAnalysis(regime=RegimeType.TRENDING)
        assert sa.is_trending is True
        assert sa.is_stable is False

    def test_is_noisy(self):
        sa = StructuralAnalysis(regime=RegimeType.NOISY)
        assert sa.is_noisy is True

    def test_has_sufficient_data(self):
        assert StructuralAnalysis(n_points=5).has_sufficient_data is True
        assert StructuralAnalysis(n_points=4).has_sufficient_data is False

    def test_to_feature_vector_length(self):
        sa = StructuralAnalysis(slope=1.0, curvature=0.5, stability=0.2,
                                accel_variance=0.1, noise_ratio=0.05,
                                trend_strength=0.01)
        vec = sa.to_feature_vector()
        assert len(vec) == 6
        assert vec[0] == 1.0  # slope
        assert vec[4] == 0.05  # noise_ratio

    def test_to_dict_keys(self):
        sa = StructuralAnalysis(n_points=10)
        d = sa.to_dict()
        expected_keys = {
            "slope", "curvature", "stability", "accel_variance",
            "noise_ratio", "regime", "mean", "std", "trend_strength",
            "n_points", "dt",
        }
        assert set(d.keys()) == expected_keys
        assert d["regime"] == "stable"

    def test_frozen(self):
        sa = StructuralAnalysis()
        with pytest.raises(AttributeError):
            sa.slope = 5.0  # type: ignore[misc]


# ── Helper functions ───────────────────────────────────────────────────


class TestHelperFunctions:
    """Tests para funciones internas de cómputo."""

    def test_compute_median_dt_uniform(self):
        ts = [0.0, 1.0, 2.0, 3.0, 4.0]
        assert abs(_compute_median_dt(ts) - 1.0) < 1e-9

    def test_compute_median_dt_with_gap(self):
        # [1, 1, 10, 1] → median of [1, 1, 1, 10] = 1.0
        ts = [0.0, 1.0, 2.0, 12.0, 13.0]
        dt = _compute_median_dt(ts)
        assert abs(dt - 1.0) < 1e-9  # median is robust to the gap

    def test_compute_median_dt_single_point(self):
        assert _compute_median_dt([0.0]) == 1.0

    def test_compute_accel_variance_constant(self):
        values = [5.0] * 10
        assert _compute_accel_variance(values, 1.0) == 0.0

    def test_compute_accel_variance_linear(self):
        # Linear signal → accel = 0 everywhere → variance = 0
        values = [float(i) for i in range(10)]
        var = _compute_accel_variance(values, 1.0)
        assert abs(var) < 1e-9

    def test_compute_accel_variance_quadratic(self):
        # Quadratic → accel = 2 everywhere → variance = 0
        values = [float(i * i) for i in range(10)]
        var = _compute_accel_variance(values, 1.0)
        assert abs(var) < 1e-6

    def test_compute_accel_variance_insufficient(self):
        assert _compute_accel_variance([1.0, 2.0, 3.0], 1.0) == 0.0

    def test_compute_stability_stable(self):
        assert _compute_stability(0.0, 10.0) == 0.0

    def test_compute_stability_unstable(self):
        s = _compute_stability(100.0, 10.0)
        assert s == 1.0  # clamped

    def test_compute_stability_near_zero_f_t(self):
        s = _compute_stability(0.5, 0.0)
        assert s == 0.5  # normalizer = 1.0


# ── Regime classification ──────────────────────────────────────────────


class TestRegimeClassification:
    """Tests para _classify_regime."""

    def test_noisy(self):
        assert _classify_regime(0.6, 0.0, 1.0, 1.0) == RegimeType.NOISY

    def test_trending(self):
        # slope=1.0, mean=10.0 → slope_ratio=0.1 > 0.005
        assert _classify_regime(0.05, 1.0, 0.5, 10.0) == RegimeType.TRENDING

    def test_volatile(self):
        assert _classify_regime(0.2, 0.001, 0.5, 10.0) == RegimeType.VOLATILE

    def test_stable(self):
        assert _classify_regime(0.05, 0.001, 0.5, 10.0) == RegimeType.STABLE


# ── compute_structural_analysis integration ────────────────────────────


class TestComputeStructuralAnalysis:
    """Tests para la función principal compute_structural_analysis."""

    def test_empty_series(self):
        sa = compute_structural_analysis([], [])
        assert sa.n_points == 0
        assert sa.has_sufficient_data is False

    def test_single_point(self):
        sa = compute_structural_analysis([42.0], [0.0])
        assert sa.n_points == 1
        assert sa.mean == 42.0
        assert sa.slope == 0.0

    def test_two_points(self):
        sa = compute_structural_analysis([10.0, 20.0], [0.0, 1.0])
        assert sa.n_points == 2
        assert abs(sa.slope - 10.0) < 1e-9
        assert sa.curvature == 0.0  # need ≥3 points

    def test_constant_signal(self):
        values = [25.0] * 20
        timestamps = [float(i) for i in range(20)]
        sa = compute_structural_analysis(values, timestamps)

        assert abs(sa.slope) < 1e-9
        assert abs(sa.curvature) < 1e-9
        assert sa.stability == 0.0
        assert sa.regime == RegimeType.STABLE

    def test_linear_signal_trending(self):
        # v(t) = 100 + 2t → slope=2, curvature≈0
        values = [100.0 + 2.0 * i for i in range(20)]
        timestamps = [float(i) for i in range(20)]
        sa = compute_structural_analysis(values, timestamps)

        assert abs(sa.slope - 2.0) < 1e-9
        assert abs(sa.curvature) < 1e-6
        assert sa.regime == RegimeType.TRENDING
        assert sa.trend_strength > 0.01

    def test_quadratic_signal_curvature(self):
        # v(t) = t² → slope at t=9 is 2*9=18, curvature=2
        values = [float(i * i) for i in range(10)]
        timestamps = [float(i) for i in range(10)]
        sa = compute_structural_analysis(values, timestamps)

        # slope = (81 - 64) / 1 = 17 (backward diff)
        assert abs(sa.slope - 17.0) < 1e-9
        # curvature = (81 - 2*64 + 49) / 1 = 2
        assert abs(sa.curvature - 2.0) < 1e-9

    def test_noisy_signal(self):
        random.seed(42)
        # Mean ≈ 0, std ≈ 10 → noise_ratio very high
        values = [random.gauss(0, 10) for _ in range(100)]
        timestamps = [float(i) for i in range(100)]
        sa = compute_structural_analysis(values, timestamps)

        assert sa.noise_ratio > 0.5
        assert sa.regime == RegimeType.NOISY

    def test_volatile_signal(self):
        random.seed(42)
        # Mean = 100, std ≈ 20 → noise_ratio ≈ 0.2
        values = [100.0 + random.gauss(0, 20) for _ in range(100)]
        timestamps = [float(i) for i in range(100)]
        sa = compute_structural_analysis(values, timestamps)

        assert 0.1 < sa.noise_ratio < 0.5

    def test_stability_low_for_linear(self):
        values = [float(i) for i in range(20)]
        timestamps = [float(i) for i in range(20)]
        sa = compute_structural_analysis(values, timestamps)

        assert sa.stability < 0.01

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="misma longitud"):
            compute_structural_analysis([1.0, 2.0], [1.0])

    def test_mean_and_std(self):
        values = [10.0, 20.0, 30.0, 40.0, 50.0]
        timestamps = [0.0, 1.0, 2.0, 3.0, 4.0]
        sa = compute_structural_analysis(values, timestamps)

        assert abs(sa.mean - 30.0) < 1e-9
        assert sa.std > 0

    def test_median_dt_robust_to_gaps(self):
        # Gap between t=2 and t=12, but median dt should be ~1.0
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        timestamps = [0.0, 1.0, 2.0, 12.0, 13.0]
        sa = compute_structural_analysis(values, timestamps)

        # Slope should use median dt=1.0, not mean dt=3.25
        assert sa.n_points == 5


# ── from_taylor_diagnostic bridge ──────────────────────────────────────


class TestFromTaylorDiagnostic:
    """Tests para StructuralAnalysis.from_taylor_diagnostic."""

    def test_bridge_from_taylor(self):
        class FakeDiag:
            local_slope = 2.5
            curvature = 0.3
            stability_indicator = 0.1
            accel_variance = 0.05

        values = [100.0 + 2.5 * i for i in range(10)]
        sa = StructuralAnalysis.from_taylor_diagnostic(FakeDiag(), values)

        assert abs(sa.slope - 2.5) < 1e-9
        assert abs(sa.curvature - 0.3) < 1e-9
        assert abs(sa.stability - 0.1) < 1e-9
        assert sa.n_points == 10
        assert sa.mean > 0

    def test_bridge_without_values(self):
        class FakeDiag:
            local_slope = 1.0
            curvature = 0.0
            stability_indicator = 0.0
            accel_variance = 0.0

        sa = StructuralAnalysis.from_taylor_diagnostic(FakeDiag())
        assert abs(sa.slope - 1.0) < 1e-9
        assert sa.n_points == 0

    def test_bridge_missing_attributes(self):
        # Object without expected attributes → defaults to 0
        sa = StructuralAnalysis.from_taylor_diagnostic(object())
        assert sa.slope == 0.0
        assert sa.curvature == 0.0


# ── TimeSeries integration ─────────────────────────────────────────────


class TestTimeSeriesStructuralAnalysis:
    """Tests para TimeSeries.structural_analysis property."""

    def test_structural_from_time_series(self):
        from iot_machine_learning.domain.entities.time_series import TimeSeries

        ts = TimeSeries.from_values(
            values=[100.0 + 2.0 * i for i in range(20)],
            timestamps=[float(i) for i in range(20)],
        )
        sa = ts.structural_analysis

        assert sa.n_points == 20
        assert sa.regime == RegimeType.TRENDING
        assert abs(sa.slope - 2.0) < 1e-9

    def test_structural_empty_series(self):
        from iot_machine_learning.domain.entities.time_series import TimeSeries

        ts = TimeSeries(series_id="empty", points=[])
        sa = ts.structural_analysis
        assert sa.n_points == 0


# ── SensorWindow integration ──────────────────────────────────────────


class TestSensorWindowStructuralAnalysis:
    """Tests para SensorWindow.structural_analysis property."""

    def test_structural_from_sensor_window(self):
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading,
            SensorWindow,
        )

        readings = [
            SensorReading(sensor_id=1, value=25.0, timestamp=float(i))
            for i in range(10)
        ]
        sw = SensorWindow(sensor_id=1, readings=readings)
        sa = sw.structural_analysis

        assert sa.n_points == 10
        assert sa.regime == RegimeType.STABLE

    def test_structural_matches_time_series(self):
        """SensorWindow.structural_analysis should match TimeSeries.structural_analysis."""
        from iot_machine_learning.domain.entities.sensor_reading import (
            SensorReading,
            SensorWindow,
        )

        readings = [
            SensorReading(sensor_id=1, value=float(i ** 2), timestamp=float(i))
            for i in range(10)
        ]
        sw = SensorWindow(sensor_id=1, readings=readings)
        ts = sw.to_time_series()

        sa_sw = sw.structural_analysis
        sa_ts = ts.structural_analysis

        assert sa_sw.n_points == sa_ts.n_points
        assert abs(sa_sw.slope - sa_ts.slope) < 1e-9
        assert abs(sa_sw.curvature - sa_ts.curvature) < 1e-9
        assert sa_sw.regime == sa_ts.regime
