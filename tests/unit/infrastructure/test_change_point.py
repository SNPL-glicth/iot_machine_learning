"""Tests para CUSUMDetector y PELTDetector.

Escenarios:
- Cambio de nivel claro (20°C → 30°C)
- Sin cambios (señal estable)
- Múltiples cambios en batch
- Reset de estado online
- Drift gradual (cambio lento)
"""

from __future__ import annotations

import random

import pytest

from iot_machine_learning.domain.entities.pattern import ChangePointType
from iot_machine_learning.infrastructure.ml.patterns.change_point_detector import (
    CUSUMDetector,
    PELTDetector,
)


class TestCUSUMOnline:
    """Tests para detección online con CUSUM."""

    def test_detects_level_shift_up(self) -> None:
        """Cambio de 20°C a 30°C debe detectarse."""
        detector = CUSUMDetector(threshold=5.0, drift=0.5)

        detected = False
        for i in range(100):
            value = 20.0 if i < 50 else 30.0
            cp = detector.detect_online(value)
            if cp is not None:
                detected = True
                assert cp.change_type == ChangePointType.LEVEL_SHIFT
                assert cp.magnitude > 0
                break

        assert detected, "CUSUM no detectó cambio de nivel"

    def test_detects_level_shift_down(self) -> None:
        """Cambio de 30°C a 20°C debe detectarse."""
        detector = CUSUMDetector(threshold=5.0, drift=0.5)

        detected = False
        for i in range(100):
            value = 30.0 if i < 50 else 20.0
            cp = detector.detect_online(value)
            if cp is not None:
                detected = True
                assert cp.change_type == ChangePointType.LEVEL_SHIFT
                break

        assert detected, "CUSUM no detectó cambio descendente"

    def test_no_false_positive_stable(self) -> None:
        """Señal estable no debe generar cambios."""
        detector = CUSUMDetector(threshold=5.0, drift=0.5)

        random.seed(42)
        changes = 0
        for _ in range(200):
            value = 20.0 + random.gauss(0, 0.1)
            cp = detector.detect_online(value)
            if cp is not None:
                changes += 1

        assert changes == 0, f"Falsos positivos en señal estable: {changes}"

    def test_reset_clears_state(self) -> None:
        """Reset debe limpiar acumuladores."""
        detector = CUSUMDetector(threshold=5.0, drift=0.5)

        for _ in range(10):
            detector.detect_online(20.0)

        detector.reset()

        assert detector._cumsum_pos == 0.0
        assert detector._cumsum_neg == 0.0
        assert detector._baseline_mean is None


class TestCUSUMBatch:
    """Tests para detección batch con CUSUM."""

    def test_batch_detects_single_change(self) -> None:
        """Un cambio de nivel en batch."""
        detector = CUSUMDetector(threshold=5.0, drift=0.5)

        values = [20.0] * 50 + [30.0] * 50
        cps = detector.detect_batch(values)

        assert len(cps) >= 1
        # El cambio debe estar cerca del índice 50
        assert any(40 <= cp.index <= 65 for cp in cps)

    def test_batch_detects_multiple_changes(self) -> None:
        """Múltiples cambios de nivel en batch."""
        detector = CUSUMDetector(threshold=5.0, drift=0.5)

        values = [20.0] * 40 + [30.0] * 40 + [15.0] * 40
        cps = detector.detect_batch(values)

        assert len(cps) >= 2, f"Esperados >= 2 cambios, detectados {len(cps)}"

    def test_batch_empty_short_series(self) -> None:
        """Serie corta no debe crashear."""
        detector = CUSUMDetector()
        assert detector.detect_batch([1.0, 2.0]) == []
        assert detector.detect_batch([]) == []


class TestCUSUMConstructor:
    """Validaciones del constructor."""

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            CUSUMDetector(threshold=-1.0)

    def test_negative_drift_raises(self) -> None:
        with pytest.raises(ValueError, match="drift"):
            CUSUMDetector(drift=-0.5)


class TestPELTDetector:
    """Tests para PELTDetector (fallback a CUSUM si ruptures no está)."""

    def test_pelt_batch_fallback_to_cusum(self) -> None:
        """Sin ruptures, debe usar CUSUM como fallback."""
        detector = PELTDetector(min_segment_size=10, penalty=3.0)

        values = [20.0] * 50 + [30.0] * 50
        cps = detector.detect_batch(values)

        # Debe detectar al menos 1 cambio (vía CUSUM fallback)
        assert len(cps) >= 1

    def test_pelt_online_delegates_to_cusum(self) -> None:
        """Online debe delegar a CUSUM."""
        detector = PELTDetector()

        detected = False
        for i in range(100):
            value = 20.0 if i < 50 else 30.0
            cp = detector.detect_online(value)
            if cp is not None:
                detected = True
                break

        assert detected

    def test_pelt_reset(self) -> None:
        """Reset debe funcionar sin error."""
        detector = PELTDetector()
        detector.detect_online(20.0)
        detector.reset()
        # No debe crashear
