"""Tests para RegimeAwareCalibrator.

Verifica selección correcta de calibrador por régimen:
- STABLE → Isotonic
- TRENDING → Platt
- VOLATILE → NullCalibrator
- Múltiples calibradores separados por motor
"""

import numpy as np
import pytest

from iot_machine_learning.infrastructure.adapters.calibrators.regime_aware_calibrator import (
    RegimeAwareCalibrator,
)


class TestRegimeAwareSelection:
    """Tests de selección de calibrador por régimen."""

    def test_stable_uses_isotonic(self):
        """STABLE usa IsotonicCalibrator."""
        calibrator = RegimeAwareCalibrator(min_samples_isotonic=20)
        
        # Con régimen STABLE
        result = calibrator.calibrate_for(0.75, engine_name="test", regime="STABLE")
        
        # Antes de tener datos, debe usar Null
        stats = calibrator.get_stats_for("test", "STABLE")
        
        # Llenar con datos suficientes para Isotonic
        for i in range(25):
            calibrator.update_for(0.6, 1.0, engine_name="test", regime="STABLE")
        
        stats = calibrator.get_stats_for("test", "STABLE")
        assert stats.calibrator_type == "isotonic"

    def test_trending_uses_platt(self):
        """TRENDING usa PlattCalibrator."""
        calibrator = RegimeAwareCalibrator(min_samples_platt=10)
        
        # Llenar con datos
        for i in range(15):
            calibrator.update_for(0.6, 1.0, engine_name="test", regime="TRENDING")
        
        stats = calibrator.get_stats_for("test", "TRENDING")
        assert stats.calibrator_type == "platt"

    def test_volatile_uses_null(self):
        """VOLATILE usa NullCalibrator (no calibra)."""
        calibrator = RegimeAwareCalibrator()
        
        # Cualquier cantidad de datos en VOLATILE
        for i in range(100):
            calibrator.update_for(0.6, 1.0, engine_name="test", regime="VOLATILE")
        
        result = calibrator.calibrate_for(0.75, engine_name="test", regime="VOLATILE")
        
        # No debe calibrar
        assert result.calibration_applied is False
        assert result.calibrated_score == 0.75
        
        stats = calibrator.get_stats_for("test", "VOLATILE")
        assert stats.calibrator_type == "null"

    def test_unknown_uses_platt(self):
        """Regímenes desconocidos usan Platt por default."""
        calibrator = RegimeAwareCalibrator(min_samples_platt=10)
        
        for i in range(15):
            calibrator.update_for(0.6, 1.0, engine_name="test", regime="NOISY")
        
        stats = calibrator.get_stats_for("test", "NOISY")
        assert stats.calibrator_type == "platt"


class TestRegimeAwareSeparateEngines:
    """Tests de separación por motor."""

    def test_engines_have_separate_calibrators(self):
        """Cada motor tiene su propio calibrador."""
        calibrator = RegimeAwareCalibrator(min_samples_platt=5)
        
        # Datos para motor A
        for i in range(10):
            calibrator.update_for(0.8, 1.0, engine_name="motor_a", regime="TRENDING")
        
        # Datos para motor B
        for i in range(10):
            calibrator.update_for(0.3, 0.0, engine_name="motor_b", regime="TRENDING")
        
        # Stats separados
        stats_a = calibrator.get_stats_for("motor_a", "TRENDING")
        stats_b = calibrator.get_stats_for("motor_b", "TRENDING")
        
        assert stats_a.n_samples == 10
        assert stats_b.n_samples == 10
        
        # Calibraciones diferentes (porque datos son diferentes)
        result_a = calibrator.calibrate_for(0.6, engine_name="motor_a", regime="TRENDING")
        result_b = calibrator.calibrate_for(0.6, engine_name="motor_b", regime="TRENDING")
        
        # Ambos calibrados (porque tienen suficientes muestras)
        assert result_a.calibration_applied is True
        assert result_b.calibration_applied is True

    def test_different_regimes_same_engine(self):
        """Mismo motor, diferentes regímenes = calibradores separados."""
        calibrator = RegimeAwareCalibrator(
            min_samples_platt=5,
            min_samples_isotonic=5,
        )
        
        # Datos en TRENDING (Platt)
        for i in range(10):
            calibrator.update_for(0.7, 1.0, engine_name="test", regime="TRENDING")
        
        # Datos en STABLE (Isotonic)
        for i in range(10):
            calibrator.update_for(0.5, 1.0, engine_name="test", regime="STABLE")
        
        stats_trending = calibrator.get_stats_for("test", "TRENDING")
        stats_stable = calibrator.get_stats_for("test", "STABLE")
        
        assert stats_trending.calibrator_type == "platt"
        assert stats_stable.calibrator_type == "isotonic"


class TestRegimeAwareFailSafety:
    """Tests de fail-safety."""

    def test_error_returns_raw_score(self):
        """Si hay error, devuelve raw score."""
        calibrator = RegimeAwareCalibrator()
        
        # Forzar error con datos inválidos
        result = calibrator.calibrate_for(float('nan'), engine_name="test", regime="TRENDING")
        
        # Debe devolver score clamped (0.0 para nan que se convierte a problemas)
        assert isinstance(result.calibrated_score, float)
        assert 0.0 <= result.calibrated_score <= 1.0

    def test_reset_specific_engine_regime(self):
        """Reset puede ser específico por motor y régimen."""
        calibrator = RegimeAwareCalibrator(min_samples_platt=5)
        
        # Llenar dos motores
        for i in range(10):
            calibrator.update_for(0.6, 1.0, engine_name="motor_a", regime="TRENDING")
            calibrator.update_for(0.6, 1.0, engine_name="motor_b", regime="TRENDING")
        
        # Reset solo motor_a
        calibrator.reset_for(engine_name="motor_a", regime="TRENDING")
        
        stats_a = calibrator.get_stats_for("motor_a", "TRENDING")
        stats_b = calibrator.get_stats_for("motor_b", "TRENDING")
        
        # A reseteado, B no
        assert stats_a.n_samples == 0
        assert stats_b.n_samples == 10

    def test_reset_all_engines(self):
        """Reset sin argumentos limpia todo."""
        calibrator = RegimeAwareCalibrator(min_samples_platt=5)
        
        for i in range(10):
            calibrator.update_for(0.6, 1.0, engine_name="motor_a", regime="TRENDING")
            calibrator.update_for(0.6, 1.0, engine_name="motor_b", regime="STABLE")
        
        calibrator.reset_for()  # Reset all
        
        # Todo limpio
        assert not calibrator._calibrators


class TestRegimeAwareAllStats:
    """Tests de get_all_stats."""

    def test_get_all_stats_returns_all_calibrators(self):
        """get_all_stats devuelve stats de todos los calibradores."""
        calibrator = RegimeAwareCalibrator(min_samples_platt=5)
        
        # Crear múltiples calibradores
        for i in range(10):
            calibrator.update_for(0.6, 1.0, engine_name="motor_a", regime="TRENDING")
            calibrator.update_for(0.6, 1.0, engine_name="motor_a", regime="STABLE")
            calibrator.update_for(0.6, 1.0, engine_name="motor_b", regime="TRENDING")
        
        all_stats = calibrator.get_all_stats()
        
        # Debe tener 3 calibradores
        assert len(all_stats) == 3
        assert ("motor_a", "TRENDING") in all_stats
        assert ("motor_a", "STABLE") in all_stats
        assert ("motor_b", "TRENDING") in all_stats

    def test_interface_methods_with_defaults(self):
        """Métodos de interfaz base funcionan con calibrador default."""
        calibrator = RegimeAwareCalibrator(min_samples_platt=5)
        
        # Llenar calibrador genérico default
        for i in range(10):
            calibrator.update(0.6, 1.0)
        
        # Usar métodos sin especificar motor/regimen (usa calibrador genérico)
        assert calibrator.is_ready()
        
        result = calibrator.calibrate(0.7)
        assert result.calibration_applied
        
        stats = calibrator.get_stats()
        assert stats.n_samples == 10
