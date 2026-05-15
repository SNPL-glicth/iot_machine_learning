"""Tests for KalmanPredictionEngine.

Verifies prediction, confidence, trend classification, sanitization,
gap detection, record_actual, and metadata completeness.
"""

from __future__ import annotations

import math
import random

import pytest

from iot_machine_learning.infrastructure.ml.engines.kalman.engine import (
    KalmanPredictionEngine,
)


class TestKalmanEngine:
    """Engine-level tests for KalmanPredictionEngine."""

    def test_can_handle_minimum_points(self) -> None:
        """can_handle returns True when n >= warmup_size."""
        engine = KalmanPredictionEngine(warmup_size=5)
        assert engine.can_handle(5) is True
        assert engine.can_handle(4) is False

    def test_warmup_size_clamped_to_minimum(self) -> None:
        """Warmup size below 3 is clamped to 3."""
        engine = KalmanPredictionEngine(warmup_size=2)
        assert engine.can_handle(3) is True
        assert engine.can_handle(2) is False

    def test_predicts_constant_signal(self) -> None:
        """Flat signal → prediction ≈ current value, high confidence."""
        engine = KalmanPredictionEngine(warmup_size=5, horizon=1)
        values = [10.0] * 20
        result = engine.predict(values)

        assert result.predicted_value == pytest.approx(10.0, abs=0.5)
        assert result.confidence >= 0.5
        assert result.trend == "stable"

    def test_predicts_linear_trend(self) -> None:
        """Linear signal → prediction extrapolates trend, v ≈ slope."""
        engine = KalmanPredictionEngine(warmup_size=5, horizon=1)
        values = [float(i) for i in range(20)]  # slope = 1
        result = engine.predict(values)

        assert result.predicted_value == pytest.approx(20.0, abs=1.0)
        assert result.trend == "up"
        # v_hat should be close to 1.0
        assert result.metadata["v_hat"] == pytest.approx(1.0, abs=0.5)

    def test_confidence_degrades_during_warmup(self) -> None:
        """Few points → confidence degraded quadratically."""
        engine = KalmanPredictionEngine(warmup_size=5)
        # 7 points: just above warmup, below 2x warmup (10)
        values = [10.0 + i * 0.1 for i in range(7)]
        result = engine.predict(values)

        # Should be degraded from max
        assert result.confidence < 0.95

    def test_confidence_low_with_high_noise(self) -> None:
        """High noise → low confidence."""
        random.seed(42)
        engine = KalmanPredictionEngine(warmup_size=5)
        # Signal ~10, noise std=10 → noise_ratio ~1.0
        values = [10.0 + random.gauss(0, 10) for _ in range(50)]
        result = engine.predict(values)

        # High noise should reduce confidence
        assert result.confidence < 0.8

    def test_metadata_includes_P_and_R(self) -> None:
        """Metadata contains Kalman diagnostic fields."""
        engine = KalmanPredictionEngine(warmup_size=5)
        values = [10.0] * 20
        result = engine.predict(values)

        assert "x_hat" in result.metadata
        assert "v_hat" in result.metadata
        assert "P_pos" in result.metadata
        assert "P_vel" in result.metadata
        assert "R" in result.metadata
        assert "Q_vel" in result.metadata

    def test_metadata_includes_confidence_interval(self) -> None:
        """supports_uncertainty = True exposes confidence_interval."""
        engine = KalmanPredictionEngine(warmup_size=5)
        values = [10.0] * 20
        result = engine.predict(values)

        assert engine.supports_uncertainty() is True
        assert "confidence_interval" in result.metadata
        lower, upper = result.metadata["confidence_interval"]
        assert lower < result.predicted_value < upper

    def test_gap_detection_warns(self) -> None:
        """Large gap in timestamps triggers gap_detected metadata."""
        engine = KalmanPredictionEngine(warmup_size=3)
        values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        # Gap of 10 between point 2 and 3 (dt should be ~1)
        timestamps = [0.0, 1.0, 2.0, 12.0, 13.0, 14.0]
        result = engine.predict(values, timestamps)

        assert result.metadata.get("gap_detected") is True

    def test_supports_uncertainty_returns_true(self) -> None:
        """Engine reports uncertainty support."""
        engine = KalmanPredictionEngine()
        assert engine.supports_uncertainty() is True

    def test_sanitizes_nan_silently(self) -> None:
        """NaN in input does not raise, returns valid PredictionResult."""
        engine = KalmanPredictionEngine(warmup_size=5)
        values = [
            1.0, 2.0, float("nan"), 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
        ]
        result = engine.predict(values)

        assert isinstance(result.predicted_value, float)
        assert math.isfinite(result.predicted_value)
        assert result.confidence >= 0.0

    def test_insufficient_data_returns_low_confidence(self) -> None:
        """Below warmup_size returns confidence=0.0."""
        engine = KalmanPredictionEngine(warmup_size=5)
        result = engine.predict([1.0, 2.0, 3.0])

        assert result.confidence == 0.0
        assert result.metadata.get("reason") == "insufficient_data"

    def test_trend_down_for_negative_slope(self) -> None:
        """Decreasing signal → trend='down'."""
        engine = KalmanPredictionEngine(warmup_size=5)
        values = [float(20 - i) for i in range(20)]
        result = engine.predict(values)

        assert result.trend == "down"
        assert result.metadata["v_hat"] < 0

    def test_record_actual_accumulates(self) -> None:
        """record_actual stores errors, recent_mae returns non-zero."""
        engine = KalmanPredictionEngine()
        for _ in range(5):
            engine.record_actual(predicted=10.0, actual=12.0)

        mae = engine.recent_mae()
        assert mae == pytest.approx(2.0, abs=1e-9)

    def test_record_actual_skips_non_finite(self) -> None:
        """Non-finite predicted/actual are silently skipped."""
        engine = KalmanPredictionEngine()
        engine.record_actual(predicted=float("nan"), actual=10.0)
        engine.record_actual(predicted=10.0, actual=float("inf"))

        assert engine.recent_mae() == 0.0

    def test_confidence_stable_on_constant_zero_signal(self) -> None:
        """Constant zero signal does not collapse confidence."""
        engine = KalmanPredictionEngine(warmup_size=5)
        values = [0.0] * 20
        result = engine.predict(values)

        assert result.confidence > 0.0
        assert result.confidence < 1.0
        assert math.isfinite(result.confidence)

    def test_repeated_predict_uses_cached_F(self) -> None:
        """F matrix is cached by dt to avoid repeated allocation."""
        from iot_machine_learning.infrastructure.ml.engines.kalman.kalman_cv_math import (
            _F_CACHE, _get_F, initialize_cv_state, predict_cv,
        )

        # Clear cache for determinism
        _F_CACHE.clear()
        state = initialize_cv_state([1.0, 2.0, 3.0, 4.0, 5.0], dt=1.0)

        for _ in range(50):
            state = predict_cv(state, dt=1.0)

        assert len(_F_CACHE) == 1
        assert _get_F(1.0) is _get_F(1.0)

    def test_multistep_process_noise_grows_cubically(self) -> None:
        """Multi-step horizon uses Q(total_dt), not h*Q(dt)."""
        values = [float(i) for i in range(20)]

        engine1 = KalmanPredictionEngine(warmup_size=5, horizon=1)
        engine2 = KalmanPredictionEngine(warmup_size=5, horizon=2)
        engine3 = KalmanPredictionEngine(warmup_size=5, horizon=3)

        r1 = engine1.predict(values)
        r2 = engine2.predict(values)
        r3 = engine3.predict(values)

        p1 = r1.metadata["P_pos"]
        p2 = r2.metadata["P_pos"]
        p3 = r3.metadata["P_pos"]

        assert p3 > p2 > p1
        # With Q(total_dt) the growth accelerates (cubic in dt),
        # so the gap h=2→3 should exceed the gap h=1→2.
        gap_1_2 = p2 - p1
        gap_2_3 = p3 - p2
        assert gap_2_3 > gap_1_2

    def test_adaptive_q_activates_after_3_innovations(self) -> None:
        """Adaptive Q turns on once at least 3 innovations are collected."""
        engine = KalmanPredictionEngine(warmup_size=5)
        # Noisy signal to force Q adaptation
        random.seed(42)
        values = [10.0 + random.gauss(0, 5.0) for _ in range(25)]

        result = engine.predict(values)

        assert result.metadata["Q_adaptive"] is True
        assert result.metadata["innovation_window"] >= 3
