"""Tests for engines/statistical_engine.py — EMA/Holt prediction."""

from __future__ import annotations

import math
import random

import pytest

from iot_machine_learning.infrastructure.ml.engines.statistical import (
    StatisticalPredictionEngine,
)


class TestStatisticalEngineBasic:

    def test_constant_signal(self) -> None:
        eng = StatisticalPredictionEngine()
        result = eng.predict([5.0] * 20)
        assert result.predicted_value == pytest.approx(5.0, abs=0.1)
        assert result.trend == "stable"

    def test_linear_signal(self) -> None:
        eng = StatisticalPredictionEngine(alpha=0.5, beta=0.3)
        values = [float(i) for i in range(20)]
        result = eng.predict(values)
        # Should predict near 20.0
        assert result.predicted_value > 18.0
        assert result.trend == "up"

    def test_downward_trend(self) -> None:
        eng = StatisticalPredictionEngine(alpha=0.5, beta=0.3)
        values = [100.0 - float(i) for i in range(20)]
        result = eng.predict(values)
        assert result.trend == "down"

    def test_confidence_range(self) -> None:
        eng = StatisticalPredictionEngine()
        result = eng.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        assert 0.0 <= result.confidence <= 1.0

    def test_finite_prediction(self) -> None:
        random.seed(42)
        eng = StatisticalPredictionEngine()
        values = [random.gauss(50, 10) for _ in range(30)]
        result = eng.predict(values)
        assert math.isfinite(result.predicted_value)


class TestStatisticalEngineMetadata:

    def test_metadata_keys(self) -> None:
        eng = StatisticalPredictionEngine()
        result = eng.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        m = result.metadata
        assert "level" in m
        assert "trend_component" in m
        assert "alpha" in m
        assert "beta" in m
        assert "residual_std" in m
        assert "diagnostic" in m

    def test_diagnostic_has_stability(self) -> None:
        eng = StatisticalPredictionEngine()
        result = eng.predict([1.0, 2.0, 3.0, 4.0, 5.0])
        diag = result.metadata["diagnostic"]
        assert "stability_indicator" in diag
        assert "local_fit_error" in diag
        assert "method" in diag
        assert diag["method"] == "ema_holt"


class TestStatisticalEngineValidation:

    def test_empty_values_raises(self) -> None:
        eng = StatisticalPredictionEngine()
        with pytest.raises(ValueError):
            eng.predict([])

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            StatisticalPredictionEngine(alpha=0.0)
        with pytest.raises(ValueError):
            StatisticalPredictionEngine(alpha=1.5)

    def test_invalid_beta_raises(self) -> None:
        with pytest.raises(ValueError):
            StatisticalPredictionEngine(beta=-0.1)

    def test_invalid_horizon_raises(self) -> None:
        with pytest.raises(ValueError):
            StatisticalPredictionEngine(horizon=0)

    def test_can_handle(self) -> None:
        eng = StatisticalPredictionEngine()
        assert eng.can_handle(3) is True
        assert eng.can_handle(2) is False

    def test_name(self) -> None:
        eng = StatisticalPredictionEngine()
        assert eng.name == "statistical_ema_holt"

    def test_supports_uncertainty(self) -> None:
        eng = StatisticalPredictionEngine()
        assert eng.supports_uncertainty() is False


class TestStatisticalEngineFallback:

    def test_insufficient_data_fallback(self) -> None:
        eng = StatisticalPredictionEngine()
        result = eng.predict([5.0, 6.0])
        assert result.metadata.get("fallback") == "insufficient_data"
        assert result.predicted_value == pytest.approx(5.5)


class TestStatisticalConfidenceRegression:
    """Regression tests for zero-mean PCA-centered confidence bug fix."""

    def test_zero_mean_centered_series_not_stuck_at_floor(self) -> None:
        """Values centered at 0 (mean ≈ 0) with high SNR must yield high confidence, not 0.20."""
        # Scenario: linear delta steps centered at 0.0
        # mean = 0.0, std ≈ 0.0345
        values = [0.01 * i - 0.055 for i in range(12)]
        assert abs(sum(values) / len(values)) < 1e-15  # strictly zero mean

        eng = StatisticalPredictionEngine(alpha=0.5, beta=0.3)
        res = eng.predict(values)
        assert res.confidence > 0.80, f"Expected high confidence for smooth trend, got {res.confidence}"
        assert res.confidence != 0.20

    def test_clean_signal_produces_high_confidence(self) -> None:
        """Low residual noise relative to signal dispersion gives high confidence near 0.95."""
        from iot_machine_learning.infrastructure.ml.engines.statistical.smoothing import (
            compute_confidence,
        )

        values = [math.sin(i * 0.2) for i in range(15)]
        mean = sum(values) / len(values)
        values_centered = [v - mean for v in values]

        conf = compute_confidence(values_centered, residual_std=0.01)
        assert conf >= 0.90, f"Expected confidence >= 0.90, got {conf}"

    def test_noisy_signal_produces_low_confidence_smoothly(self) -> None:
        """Noisy signal where residual std approaches or exceeds signal std degrades confidence to floor."""
        from iot_machine_learning.infrastructure.ml.engines.statistical.smoothing import (
            compute_confidence,
        )

        values = [0.01 * ((-1) ** i) for i in range(12)]
        signal_std = math.sqrt(sum(v ** 2 for v in values) / len(values))
        # When residual_std == signal_std (unexplained variance ratio = 1.0)
        conf = compute_confidence(values, residual_std=signal_std)
        assert conf == pytest.approx(0.20), f"Expected floor confidence 0.20, got {conf}"

        # When residual_std is moderate (ratio = 0.5)
        conf_mod = compute_confidence(values, residual_std=signal_std * 0.5)
        assert 0.45 <= conf_mod <= 0.55, f"Expected confidence ~0.50, got {conf_mod}"

    def test_zero_variance_constant_signal_edge_case(self) -> None:
        """Constant signal (std = 0) with zero residuals yields max confidence without division error."""
        from iot_machine_learning.infrastructure.ml.engines.statistical.smoothing import (
            compute_confidence,
        )

        values = [0.0] * 12
        conf_zero_res = compute_confidence(values, residual_std=0.0)
        assert conf_zero_res == pytest.approx(0.95)

        # Constant signal with nonzero residual yields floor
        conf_noisy_const = compute_confidence(values, residual_std=0.05)
        assert conf_noisy_const == pytest.approx(0.20)

    def test_expert_adapter_pca_integration(self) -> None:
        """Integration with StatisticalExpertAdapter over synthetic trajectory."""
        import numpy as np
        from infrastructure.ml.adapters.statistical_adapter import StatisticalExpertAdapter
        from domain.entities.rosa_roja.movement import Movement
        from domain.entities.rosa_roja.trajectory import Trajectory, TerminalState

        rng = np.random.default_rng(42)
        # Smooth directional trend in delta_states
        deltas = np.zeros((12, 10))
        for i in range(12):
            deltas[i, 0] = 0.005 * i  # smooth trend in first feature
            deltas[i, 1:] = rng.normal(0, 0.0005, size=9)

        movements = [Movement.from_raw(deltas[i], delta_time=1.0, timestamp=float(i)) for i in range(12)]
        traj = Trajectory(
            movements=tuple(movements),
            coherence_score=0.03,
            invalidation_step=None,
            terminal_state=TerminalState(state_vector=deltas[-1], step_index=11, confidence=0.03),
        )

        adapter = StatisticalExpertAdapter(StatisticalPredictionEngine())
        score = adapter.evaluate_trajectory(traj)
        assert score > 0.60, f"Expected adapter confidence > 0.60 on smooth trend, got {score}"
