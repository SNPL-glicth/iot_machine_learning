"""Tests para CUSUMDetector cooldown (FIX-6).

Suprime detecciones oscilantes después de un change point detectado.
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.patterns.change_point_detector import (
    CUSUMDetector,
)


class TestCUSUMCooldown:
    """Cooldown suppresses oscillation after detection."""

    def test_cooldown_suppresses_oscillation(self) -> None:
        """After detection, next N calls return None despite drift."""
        detector = CUSUMDetector(threshold=3.0, drift=0.1, cooldown_periods=3)

        # Phase 1: stable baseline
        for _ in range(5):
            detector.detect_online(10.0)

        # Phase 2: strong drift → should detect
        result = None
        for i in range(20):
            r = detector.detect_online(20.0)
            if r is not None:
                result = r
                break

        assert result is not None

        # Phase 3: during cooldown, even with drift → None
        for _ in range(3):
            r = detector.detect_online(25.0)
            assert r is None, "Cooldown should suppress detection"

        # Phase 4: after cooldown, detection resumes
        result2 = None
        for i in range(20):
            r = detector.detect_online(30.0)
            if r is not None:
                result2 = r
                break

        assert result2 is not None

    def test_cooldown_zero_disables_suppression(self) -> None:
        """cooldown_periods=0: old behavior, no suppression."""
        detector = CUSUMDetector(threshold=3.0, drift=0.1, cooldown_periods=0)

        for _ in range(5):
            detector.detect_online(10.0)

        # First detection
        result1 = None
        for _ in range(20):
            r = detector.detect_online(20.0)
            if r is not None:
                result1 = r
                break
        assert result1 is not None

        # Immediately can detect again (no cooldown)
        result2 = None
        for _ in range(20):
            r = detector.detect_online(30.0)
            if r is not None:
                result2 = r
                break
        assert result2 is not None

    def test_reset_clears_cooldown(self) -> None:
        """reset() clears cooldown state."""
        detector = CUSUMDetector(threshold=3.0, drift=0.1, cooldown_periods=5)

        for _ in range(5):
            detector.detect_online(10.0)

        result = None
        for _ in range(20):
            r = detector.detect_online(20.0)
            if r is not None:
                result = r
                break
        assert result is not None
        assert detector._cooldown_remaining == 5

        detector.reset()
        assert detector._cooldown_remaining == 0

    def test_invalid_cooldown_raises(self) -> None:
        """cooldown_periods < 0 raises ValueError."""
        with pytest.raises(ValueError, match="cooldown_periods"):
            CUSUMDetector(cooldown_periods=-1)
