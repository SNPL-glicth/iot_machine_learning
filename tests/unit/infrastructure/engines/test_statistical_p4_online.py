"""Tests para StatisticalPredictionEngine P4: online alpha adjustment.

Verifica:
1. record_actual() ajusta alpha incrementalmente
2. Re-optimización deferred ahora se marca cada 20 calls (no 50)
3. Alpha se clampa en [0.05, 0.95]
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.engines.statistical import (
    StatisticalPredictionEngine,
)


class TestStatisticalOnlineAlpha:
    """P4: online alpha micro-adjustment."""

    def test_alpha_increases_on_large_error(self) -> None:
        """Gran error → alpha sube (más reactividad)."""
        engine = StatisticalPredictionEngine(alpha=0.3, enable_optimization=True)
        initial_alpha = engine._alpha
        # Error grande relativo al valor actual
        engine.record_actual(predicted=10.0, actual=50.0)
        assert engine._alpha > initial_alpha, (
            f"Expected alpha to increase after large error: {initial_alpha} → {engine._alpha}"
        )

    def test_alpha_decreases_on_small_error(self) -> None:
        """Pequeño error → alpha baja (más suavizado)."""
        engine = StatisticalPredictionEngine(alpha=0.5, enable_optimization=True)
        initial_alpha = engine._alpha
        # Error pequeño (< 10% del scale)
        engine.record_actual(predicted=100.0, actual=101.0)
        assert engine._alpha < initial_alpha, (
            f"Expected alpha to decrease after small error: {initial_alpha} → {engine._alpha}"
        )

    def test_alpha_clamped_to_maximum(self) -> None:
        """Alpha no debe exceder 0.95."""
        engine = StatisticalPredictionEngine(alpha=0.94, enable_optimization=True)
        for _ in range(20):
            engine.record_actual(predicted=0.0, actual=100.0)
        assert engine._alpha <= 0.95, f"Alpha exceeded max clamp: {engine._alpha}"

    def test_alpha_clamped_to_minimum(self) -> None:
        """Alpha no debe ser menor a 0.05."""
        engine = StatisticalPredictionEngine(alpha=0.06, enable_optimization=True)
        for _ in range(20):
            engine.record_actual(predicted=100.0, actual=100.0)
        assert engine._alpha >= 0.05, f"Alpha fell below min clamp: {engine._alpha}"

    def test_reoptimization_flag_every_20_calls(self) -> None:
        """_needs_reoptimization se marca tras 20 record_actual()."""
        engine = StatisticalPredictionEngine(enable_optimization=True)
        for i in range(19):
            engine.record_actual(predicted=50.0, actual=50.0)
        assert engine._needs_reoptimization is False
        engine.record_actual(predicted=50.0, actual=50.0)
        assert engine._needs_reoptimization is True

    def test_no_adjustment_when_optimization_disabled(self) -> None:
        """Con enable_optimization=False, alpha no cambia."""
        engine = StatisticalPredictionEngine(alpha=0.3, enable_optimization=False)
        engine.record_actual(predicted=10.0, actual=50.0)
        assert engine._alpha == pytest.approx(0.3, abs=1e-6)

    def test_optimize_resets_counter(self) -> None:
        """Llamar optimize() resetea _needs_reoptimization y _prediction_count."""
        engine = StatisticalPredictionEngine(enable_optimization=True)
        for _ in range(20):
            engine.record_actual(predicted=50.0, actual=50.0)
        assert engine._needs_reoptimization is True
        assert engine._prediction_count == 20
        engine.optimize()
        assert engine._needs_reoptimization is False
        assert engine._prediction_count == 0
