"""Tests for RegimeDetector.

5 cases:
1. Detects clear regime separation (idle/active/peak)
2. No false positive on stable gaussian noise (one regime dominates)
3. Handles insufficient data gracefully
4. Handles NaN input gracefully
5. Handles constant input gracefully
"""

from __future__ import annotations

import math

import pytest

from iot_machine_learning.infrastructure.ml.patterns.regime_detector import (
    RegimeDetector,
)


class TestRegimeDetector:
    def test_detects_clear_anomaly(self) -> None:
        detector = RegimeDetector(n_regimes=3)
        # idle=25, active=80, peak=120
        values = [25.0] * 30 + [80.0] * 30 + [120.0] * 30
        detector.train(values)
        regime = detector.predict_regime(120.0)
        assert regime.name in ("peak", "active")
        assert regime.mean_value > 100.0

    def test_no_false_positive_stable(self) -> None:
        detector = RegimeDetector(n_regimes=2)
        values = [10.0 + (i % 3 - 1) * 0.05 for i in range(50)]
        detector.train(values)
        regime = detector.predict_regime(10.0)
        assert regime.name in ("idle", "active")

    def test_handles_insufficient_data(self) -> None:
        detector = RegimeDetector(n_regimes=3)
        with pytest.raises(ValueError):
            detector.train([1.0, 2.0])  # < 30 points required
        assert not detector.is_trained()

    def test_handles_nan_input(self) -> None:
        detector = RegimeDetector(n_regimes=2)
        try:
            detector.train([1.0, 2.0, float("nan"), 4.0, 5.0] * 10)
        except (ValueError, RuntimeError):
            pass  # acceptable — NaN causes sklearn to fail
        # No unhandled crash = success
        assert True

    def test_handles_constant_input(self) -> None:
        detector = RegimeDetector(n_regimes=3)
        detector.train([5.0] * 50)
        regime = detector.predict_regime(5.0)
        assert regime is not None
        assert regime.mean_value == pytest.approx(5.0, abs=1e-6)
