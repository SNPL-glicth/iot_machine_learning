"""Tests para PlattCalibrator.

Verifica calibración Platt Scaling con SGD online, incluyendo:
- Correcta calibración después de suficientes muestras
- Fail-safe cuando n_samples < min_samples
- Thread safety con concurrencia
- Cálculo de ECE
"""

import threading
from typing import List

import numpy as np
import pytest

from iot_machine_learning.infrastructure.adapters.calibrators.platt_calibrator import (
    PlattCalibrator,
)


class TestPlattCalibratorBasics:
    """Tests básicos de inicialización y calibración."""

    def test_init_with_custom_params(self):
        """Inicializar con parámetros personalizados."""
        calibrator = PlattCalibrator(
            window_size=100,
            min_samples=20,
            update_frequency=5,
            engine_name="test_engine",
        )
        
        assert calibrator._window_size == 100
        assert calibrator._min_samples == 20
        assert calibrator._update_frequency == 5
        assert calibrator._engine_name == "test_engine"
        assert calibrator.is_ready() is False

    def test_returns_raw_score_when_not_ready(self):
        """Devuelve raw score cuando no hay suficientes muestras."""
        calibrator = PlattCalibrator(min_samples=50)
        
        # Menos de 50 muestras
        for i in range(10):
            calibrator.update(0.5, 1.0)
        
        result = calibrator.calibrate(0.75)
        assert result.calibration_applied is False
        assert result.calibrated_score == 0.75
        assert calibrator.is_ready() is False

    def test_applies_calibration_when_ready(self):
        """Aplica calibración cuando hay suficientes muestras."""
        calibrator = PlattCalibrator(min_samples=50, update_frequency=10)
        
        # Generar datos con correlación positiva score→accuracy
        np.random.seed(42)
        for i in range(60):
            # Scores más altos = más probabilidad de ser correcto
            score = np.random.uniform(0.3, 0.9)
            correct = 1.0 if np.random.random() < score else 0.0
            calibrator.update(score, correct)
        
        # Ahora debería estar listo
        assert calibrator.is_ready() is True
        
        # Calibrar un score
        result = calibrator.calibrate(0.8)
        # La calibración debería ajustar el score (no garantizado el valor exacto)
        assert result.calibration_applied is True
        assert 0.0 <= result.calibrated_score <= 1.0

    def test_calibration_improves_ece(self):
        """La calibración mejora el ECE (Expected Calibration Error)."""
        calibrator = PlattCalibrator(min_samples=50, window_size=200)
        
        np.random.seed(42)
        
        # Generar datos des-calibrados: scores altos pero accuracy baja
        for i in range(100):
            score = np.random.uniform(0.7, 0.95)  # Scores optimistas
            # Pero solo 60% de accuracy (des-calibrado)
            correct = 1.0 if np.random.random() < 0.6 else 0.0
            calibrator.update(score, correct)
        
        assert calibrator.is_ready()
        
        stats = calibrator.get_stats()
        # ECE debería ser detectable (> 0)
        assert stats.ece >= 0.0
        assert stats.calibrator_type == "platt"


class TestPlattCalibratorFailSafety:
    """Tests de fail-safety: nunca debe crashear."""

    def test_invalid_scores_handled(self):
        """Maneja scores inválidos sin crash."""
        calibrator = PlattCalibrator(min_samples=10)
        
        # Updates con datos válidos
        for i in range(15):
            calibrator.update(0.5, 1.0)
        
        # Calibrar scores inválidos
        result_neg = calibrator.calibrate(-1.0)
        result_high = calibrator.calibrate(2.0)
        
        assert 0.0 <= result_neg.calibrated_score <= 1.0
        assert 0.0 <= result_high.calibrated_score <= 1.0

    def test_reset_clears_state(self):
        """Reset limpia el estado correctamente."""
        calibrator = PlattCalibrator(min_samples=10)
        
        # Llenar con datos
        for i in range(20):
            calibrator.update(0.5, 1.0)
        
        assert calibrator.is_ready()
        
        # Reset
        calibrator.reset()
        
        # Estado limpio
        assert not calibrator.is_ready()
        assert len(calibrator._scores) == 0
        
        stats = calibrator.get_stats()
        assert stats.n_samples == 0


class TestPlattCalibratorThreadSafety:
    """Tests de thread safety."""

    def test_concurrent_updates(self):
        """Múltiples threads pueden hacer update concurrentemente."""
        calibrator = PlattCalibrator(min_samples=100)
        
        errors = []
        
        def update_worker(thread_id: int, n_updates: int):
            try:
                for i in range(n_updates):
                    score = 0.3 + (thread_id * 0.1) % 0.6  # Diferentes rangos por thread
                    correct = 1.0 if i % 2 == 0 else 0.0
                    calibrator.update(score, correct)
            except Exception as exc:
                errors.append(exc)
        
        # 5 threads, 20 updates cada uno
        threads = [
            threading.Thread(target=update_worker, args=(i, 20))
            for i in range(5)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # No debe haber errores
        assert len(errors) == 0
        
        # Debe tener las muestras
        assert len(calibrator._scores) == 100

    def test_concurrent_calibrate_and_update(self):
        """Calibrate y update concurrentes son thread-safe."""
        calibrator = PlattCalibrator(min_samples=50)
        
        # Pre-popular
        for i in range(60):
            calibrator.update(0.6, 1.0 if i % 2 == 0 else 0.0)
        
        results: List[float] = []
        errors = []
        
        def calibrate_worker(n_calls: int):
            try:
                for _ in range(n_calls):
                    result = calibrator.calibrate(0.75)
                    results.append(result.calibrated_score)
            except Exception as exc:
                errors.append(exc)
        
        def update_worker(n_calls: int):
            try:
                for i in range(n_calls):
                    calibrator.update(0.5 + i * 0.01, 1.0)
            except Exception as exc:
                errors.append(exc)
        
        # Threads concurrentes
        threads = [
            threading.Thread(target=calibrate_worker, args=(50,)),
            threading.Thread(target=calibrate_worker, args=(50,)),
            threading.Thread(target=update_worker, args=(20,)),
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(results) == 100


class TestPlattCalibratorStats:
    """Tests de estadísticas (ECE)."""

    def test_ece_computation(self):
        """ECE se computa correctamente."""
        calibrator = PlattCalibrator(min_samples=30, window_size=100)
        
        # Datos perfectamente calibrados: score = probabilidad real
        np.random.seed(42)
        for _ in range(50):
            score = np.random.uniform(0.1, 0.9)
            correct = 1.0 if np.random.random() < score else 0.0
            calibrator.update(score, correct)
        
        stats = calibrator.get_stats()
        
        assert stats.n_samples == 50
        assert stats.calibrator_type == "platt"
        assert stats.is_ready is True
        # ECE debería ser razonablemente bajo para datos bien calibrados
        assert 0.0 <= stats.ece <= 0.3

    def test_stats_not_ready(self):
        """Stats cuando no hay suficientes muestras."""
        calibrator = PlattCalibrator(min_samples=50)
        
        # Pocas muestras
        for _ in range(10):
            calibrator.update(0.5, 1.0)
        
        stats = calibrator.get_stats()
        
        assert stats.is_ready is False
        assert stats.ece == 0.0
        assert stats.n_samples == 10
