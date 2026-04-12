"""Tests para IsotonicCalibrator.

Verifica calibración Isotonic Regression, incluyendo:
- Correcta calibración monótona después de suficientes muestras
- Manejo de datos con relación no-lineal score→probabilidad
- Thread safety
"""

import threading
from typing import List

import numpy as np
import pytest

from iot_machine_learning.infrastructure.adapters.calibrators.isotonic_calibrator import (
    IsotonicCalibrator,
)


class TestIsotonicCalibratorBasics:
    """Tests básicos."""

    def test_init_with_custom_params(self):
        """Inicializar con parámetros personalizados."""
        calibrator = IsotonicCalibrator(
            window_size=500,
            min_samples=100,
            engine_name="test_engine",
        )
        
        assert calibrator._window_size == 500
        assert calibrator._min_samples == 100
        assert calibrator._engine_name == "test_engine"

    def test_returns_raw_score_when_not_ready(self):
        """Devuelve raw score cuando no hay suficientes muestras."""
        calibrator = IsotonicCalibrator(min_samples=200)
        
        # Menos de 200 muestras
        for i in range(50):
            calibrator.update(0.5, 1.0)
        
        result = calibrator.calibrate(0.75)
        assert result.calibration_applied is False
        assert result.calibrated_score == 0.75
        assert calibrator.is_ready() is False

    def test_applies_calibration_when_ready(self):
        """Aplica calibración cuando hay suficientes muestras."""
        calibrator = IsotonicCalibrator(min_samples=50)  # Reducido para test
        
        # Generar datos monótonos crecientes
        np.random.seed(42)
        scores = np.linspace(0.1, 0.9, 60)
        for score in scores:
            # Probabilidad de acierto = score (monótona)
            correct = 1.0 if np.random.random() < score else 0.0
            calibrator.update(float(score), correct)
        
        # Forzar recalibración
        for _ in range(20):
            calibrator.update(0.5, 1.0)
        
        assert calibrator.is_ready() is True
        
        # La función isotónica debe preservar monotonicidad
        result_low = calibrator.calibrate(0.2)
        result_high = calibrator.calibrate(0.8)
        
        assert result_low.calibration_applied is True
        assert result_high.calibration_applied is True
        # Calibrado alto > calibrado bajo (monótono)
        assert result_high.calibrated_score > result_low.calibrated_score


class TestIsotonicCalibratorMonotonicity:
    """Tests de propiedad monótona de Isotonic Regression."""

    def test_preserves_monotonicity(self):
        """La calibración preserva orden monótono."""
        calibrator = IsotonicCalibrator(min_samples=50)
        
        # Datos donde scores bajos = baja accuracy, scores altos = alta accuracy
        np.random.seed(42)
        for i in range(60):
            score = 0.2 + (i / 60.0) * 0.6  # 0.2 a 0.8
            # Accuracy proporcional al score
            correct_prob = 0.3 + (score - 0.2) / 0.6 * 0.6  # 0.3 a 0.9
            correct = 1.0 if np.random.random() < correct_prob else 0.0
            calibrator.update(score, correct)
        
        # Forzar fit
        for _ in range(20):
            calibrator.update(0.5, 1.0)
        
        # Test monotonicidad
        scores_test = [0.25, 0.40, 0.55, 0.70, 0.85]
        calibrated_values = [
            calibrator.calibrate(s).calibrated_score for s in scores_test
        ]
        
        # Cada valor debe ser >= anterior (monótono creciente)
        for i in range(1, len(calibrated_values)):
            assert calibrated_values[i] >= calibrated_values[i-1] - 1e-6  # Tolerancia numérica


class TestIsotonicCalibratorFailSafety:
    """Tests de fail-safety."""

    def test_handles_few_samples(self):
        """Maneja pocas muestras sin crash."""
        calibrator = IsotonicCalibrator(min_samples=200)
        
        # Solo 10 muestras
        for i in range(10):
            calibrator.update(0.5, 1.0)
        
        result = calibrator.calibrate(0.6)
        assert result.calibration_applied is False
        assert not calibrator.is_ready()

    def test_reset_clears_state(self):
        """Reset limpia correctamente."""
        calibrator = IsotonicCalibrator(min_samples=50)
        
        for i in range(60):
            calibrator.update(0.5, 1.0)
        
        assert calibrator.is_ready()
        
        calibrator.reset()
        
        assert not calibrator.is_ready()
        assert len(calibrator._scores) == 0
        assert len(calibrator._isotonic_function) == 0


class TestIsotonicCalibratorStats:
    """Tests de estadísticas."""

    def test_ece_computation(self):
        """ECE computado correctamente."""
        calibrator = IsotonicCalibrator(min_samples=50, window_size=150)
        
        np.random.seed(42)
        # Generar 100 puntos con relación monótona
        for i in range(100):
            score = 0.1 + (i / 100.0) * 0.8
            correct_prob = score
            correct = 1.0 if np.random.random() < correct_prob else 0.0
            calibrator.update(score, correct)
        
        # Forzar fit
        for _ in range(20):
            calibrator.update(0.5, 1.0)
        
        stats = calibrator.get_stats()
        
        assert stats.n_samples == 120  # 100 + 20
        assert stats.calibrator_type == "isotonic"
        assert stats.is_ready is True
        assert 0.0 <= stats.ece <= 0.5  # ECE razonable
        assert len(stats.reliability) > 0  # Debe tener bins


class TestIsotonicCalibratorThreadSafety:
    """Tests de thread safety."""

    def test_concurrent_updates(self):
        """Updates concurrentes son seguros."""
        calibrator = IsotonicCalibrator(min_samples=50)
        
        errors = []
        
        def update_worker(thread_id: int, n_updates: int):
            try:
                for i in range(n_updates):
                    score = 0.3 + (i % 50) / 100.0
                    correct = 1.0 if i % 2 == 0 else 0.0
                    calibrator.update(score, correct)
            except Exception as exc:
                errors.append(exc)
        
        threads = [
            threading.Thread(target=update_worker, args=(i, 30))
            for i in range(3)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(calibrator._scores) == 90  # 3 threads * 30
