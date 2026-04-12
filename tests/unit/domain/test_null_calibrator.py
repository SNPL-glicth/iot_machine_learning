"""Tests para NullCalibrator.

Verifica que el calibrador no-op funciona correctamente como
fallback cuando la calibración está desactivada.
"""

import pytest

from iot_machine_learning.domain.ports.confidence_calibrator_port import (
    CalibrationStats,
    CalibratedScore,
    NullCalibrator,
)


class TestNullCalibrator:
    """Tests de NullCalibrator (no-op)."""

    def test_calibrate_returns_raw_score(self):
        """calibrate() devuelve el mismo score sin modificar."""
        calibrator = NullCalibrator()
        
        # Test con score válido
        result = calibrator.calibrate(0.75)
        assert result.raw_score == 0.75
        assert result.calibrated_score == 0.75
        assert result.calibration_applied is False
        assert result.calibration_delta == 0.0

    def test_calibrate_clamps_to_valid_range(self):
        """calibrate() clamp a [0, 1] para scores fuera de rango."""
        calibrator = NullCalibrator()
        
        # Score negativo
        result = calibrator.calibrate(-0.5)
        assert result.calibrated_score == 0.0
        
        # Score > 1
        result = calibrator.calibrate(1.5)
        assert result.calibrated_score == 1.0

    def test_update_is_no_op(self):
        """update() no hace nada (no falla)."""
        calibrator = NullCalibrator()
        
        # No debe lanzar excepción
        calibrator.update(0.5, 1.0)
        calibrator.update(0.8, 0.0)
        calibrator.update(1.2, 1.0)  # Score inválido
        
        # Verificar que n_samples sigue siendo 0
        stats = calibrator.get_stats()
        assert stats.n_samples == 0

    def test_is_ready_always_false(self):
        """is_ready() siempre devuelve False."""
        calibrator = NullCalibrator()
        
        assert calibrator.is_ready() is False
        
        # Incluso después de updates
        calibrator.update(0.5, 1.0)
        assert calibrator.is_ready() is False

    def test_get_stats_returns_empty(self):
        """get_stats() devuelve stats vacías."""
        calibrator = NullCalibrator()
        
        stats = calibrator.get_stats()
        
        assert isinstance(stats, CalibrationStats)
        assert stats.n_samples == 0
        assert stats.ece == 0.0
        assert stats.is_ready is False
        assert stats.calibrator_type == "null"
        assert stats.reliability == {}

    def test_reset_is_no_op(self):
        """reset() no hace nada (no falla)."""
        calibrator = NullCalibrator()
        
        # No debe lanzar excepción
        calibrator.reset()
        
        # Estado igual
        assert calibrator.is_ready() is False

    def test_multiple_calls_independent(self):
        """Múltiples llamadas son independientes y no acumulan estado."""
        calibrator = NullCalibrator()
        
        # 100 llamadas
        for i in range(100):
            score = i / 100.0
            result = calibrator.calibrate(score)
            calibrator.update(score, 1.0 if i % 2 == 0 else 0.0)
            
            assert result.calibrated_score == score
            assert result.calibration_applied is False
        
        # Stats deben seguir siendo vacías
        stats = calibrator.get_stats()
        assert stats.n_samples == 0
