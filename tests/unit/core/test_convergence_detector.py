"""Tests for core/convergence_detector.py."""

import pytest

from core.tuning.convergence_detector import (
    ConvergenceDetector,
    ConvergenceStatus,
    ConvergenceResult,
)


class TestConvergenceDetector:
    def test_init_default_params(self):
        detector = ConvergenceDetector()
        assert detector._window == 20
        assert detector._convergence_threshold == 1e-4
        assert detector._oscillation_threshold == 3
        assert detector._divergence_factor == 2.0

    def test_insufficient_data_initial(self):
        detector = ConvergenceDetector()
        result = detector.update(0.5)
        assert result.status == ConvergenceStatus.INSUFFICIENT_DATA
        assert result.current_value == 0.5
        assert result.recommendation == "need_more_data"

    def test_converging_sequence(self):
        detector = ConvergenceDetector(window=10, convergence_threshold=0.01)
        values = [1.0, 0.95, 0.92, 0.90, 0.88, 0.87, 0.86, 0.855, 0.85, 0.848]
        result = None
        for val in values:
            result = detector.update(val)
        assert result.status == ConvergenceStatus.CONVERGING

    def test_converged_sequence(self):
        detector = ConvergenceDetector(window=10, convergence_threshold=0.001)
        # Stable values with tiny changes
        values = [0.5, 0.5001, 0.5002, 0.5001, 0.5003, 0.5002, 0.5001, 0.5002]
        result = None
        for val in values:
            result = detector.update(val)
        assert result.status == ConvergenceStatus.CONVERGED
        assert result.steps_since_change >= 5

    def test_oscillating_sequence(self):
        detector = ConvergenceDetector(window=10, oscillation_threshold=3)
        # Alternating up/down
        values = [0.5, 0.6, 0.4, 0.7, 0.3, 0.8, 0.2, 0.9]
        result = None
        for val in values:
            result = detector.update(val)
        assert result.status == ConvergenceStatus.OSCILLATING
        assert result.oscillation_count >= 3

    def test_diverging_sequence(self):
        detector = ConvergenceDetector(window=15, divergence_factor=2.0)
        # Increasing deltas
        values = [1.0, 1.01, 1.03, 1.06, 1.10, 1.15, 1.21, 1.28, 1.36, 1.45, 1.55, 1.66]
        result = None
        for val in values:
            result = detector.update(val)
        assert result.status == ConvergenceStatus.DIVERGING

    def test_reset_clears_history(self):
        detector = ConvergenceDetector()
        detector.update(0.5)
        detector.update(0.6)
        detector.reset()
        assert len(detector.get_history()) == 0

    def test_get_history_returns_values(self):
        detector = ConvergenceDetector()
        values = [0.1, 0.2, 0.3]
        for val in values:
            detector.update(val)
        history = detector.get_history()
        assert history == values

    def test_convergence_threshold_respected(self):
        detector = ConvergenceDetector(convergence_threshold=0.1)
        # Changes below threshold
        values = [1.0, 1.05, 1.08, 1.09, 1.095, 1.098, 1.099, 1.0995]
        result = None
        for val in values:
            result = detector.update(val)
        assert result.steps_since_change > 0

    def test_oscillation_count_accurate(self):
        detector = ConvergenceDetector(window=10)
        # Exactly 4 sign changes
        values = [0.5, 0.6, 0.55, 0.65, 0.6, 0.7]
        result = None
        for val in values:
            result = detector.update(val)
        assert result.oscillation_count == 4

    def test_window_size_respected(self):
        detector = ConvergenceDetector(window=5)
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        for val in values:
            detector.update(val)
        history = detector.get_history()
        assert len(history) == 5  # Window size

    def test_single_value_insufficient(self):
        detector = ConvergenceDetector()
        result = detector.update(1.0)
        assert result.status == ConvergenceStatus.INSUFFICIENT_DATA
        assert result.delta_mean == 0.0
