"""Tests para calibración de ensemble."""
import pytest
import numpy as np

from core.ensemble.ensemble_calibrator import (
    EnsembleCalibrator,
    DetectionRateProfile,
    DetectionRateMeasurer,
    CalibratedWeights,
)
from core.parameters.numerical_constants import STAT_THRESHOLDS


class MockDetector:
    """Detector mock para testing."""
    
    def __init__(self, anomaly_rate: float = 0.01):
        self.anomaly_rate = anomaly_rate
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Retorna predicciones con tasa de anomalía especificada."""
        n_anomalies = int(len(data) * self.anomaly_rate)
        predictions = np.zeros(len(data))
        predictions[:n_anomalies] = 1.0
        np.random.shuffle(predictions)
        return predictions


def test_calibration_reduces_high_rate_detectors():
    """Detector con alta tasa debe tener peso reducido."""
    calibrator = EnsembleCalibrator()
    
    raw_weights = {'detector_a': 0.5, 'detector_b': 0.5}
    
    profiles = [
        DetectionRateProfile(
            detector_name='detector_a',
            expected_rate=0.01,  # 1% esperado
            empirical_rate=0.10,  # 10% real → 10× más
            samples_evaluated=1000
        ),
        DetectionRateProfile(
            detector_name='detector_b',
            expected_rate=0.01,
            empirical_rate=0.01,  # Normal
            samples_evaluated=1000
        ),
    ]
    
    result = calibrator.calibrate_by_detection_rate(raw_weights, profiles)
    
    # detector_a debe tener peso reducido ~10×
    assert result.calibrated_weights['detector_a'] < raw_weights['detector_a']
    assert result.calibrated_weights['detector_b'] > result.calibrated_weights['detector_a']
    
    # Suma debe ser 1.0
    assert result.validate()


def test_calibration_increases_low_rate_detectors():
    """Detector con baja tasa debe tener peso aumentado."""
    calibrator = EnsembleCalibrator()
    
    raw_weights = {'detector_a': 0.5, 'detector_b': 0.5}
    
    profiles = [
        DetectionRateProfile(
            detector_name='detector_a',
            expected_rate=0.01,
            empirical_rate=0.001,  # 0.1% real → 10× menos
            samples_evaluated=1000
        ),
        DetectionRateProfile(
            detector_name='detector_b',
            expected_rate=0.01,
            empirical_rate=0.01,  # Normal
            samples_evaluated=1000
        ),
    ]
    
    result = calibrator.calibrate_by_detection_rate(raw_weights, profiles)
    
    # detector_a debe tener peso aumentado ~10×
    assert result.calibrated_weights['detector_a'] > raw_weights['detector_a']
    assert result.calibrated_weights['detector_a'] > result.calibrated_weights['detector_b']
    
    # Suma debe ser 1.0
    assert result.validate()


def test_expected_rates_match_thresholds():
    """Tasas esperadas deben coincidir con thresholds configurados."""
    calibrator = EnsembleCalibrator()
    rates = calibrator.compute_expected_rates()
    
    # Z-score con threshold=3.0 debe detectar ~0.27% por cola
    assert 0.002 < rates['z_score'] < 0.01  # ~0.27% * 2 colas
    
    # Contamination debe coincidir
    assert rates['isolation_forest'] == STAT_THRESHOLDS.CONTAMINATION_DEFAULT
    
    # IQR debe ser ~0.7%
    assert abs(rates['iqr'] - 0.007) < 0.001


def test_detection_rate_measurer():
    """DetectionRateMeasurer debe medir tasas empíricas correctamente."""
    measurer = DetectionRateMeasurer()
    
    # Crear datos normales
    np.random.seed(42)
    data = np.random.randn(2000)
    
    # Crear detectores mock
    detectors = {
        'low_rate': MockDetector(anomaly_rate=0.005),
        'high_rate': MockDetector(anomaly_rate=0.05),
    }
    
    profiles = measurer.measure_rates(detectors, data, min_samples=100)
    
    assert len(profiles) == 2
    assert profiles[0].detector_name in ['low_rate', 'high_rate']
    assert profiles[1].detector_name in ['low_rate', 'high_rate']
    assert profiles[0].samples_evaluated == 2000
    assert profiles[1].samples_evaluated == 2000


def test_detection_rate_measurer_insufficient_data():
    """Debe lanzar error con datos insuficientes."""
    measurer = DetectionRateMeasurer()
    
    detectors = {'detector': MockDetector()}
    data = np.array([1.0, 2.0, 3.0])  # Solo 3 samples
    
    with pytest.raises(ValueError, match="Datos insuficientes"):
        measurer.measure_rates(detectors, data, min_samples=100)


def test_rate_ratio_calculation():
    """Rate ratio debe calcularse correctamente."""
    profile = DetectionRateProfile(
        detector_name='test',
        expected_rate=0.01,
        empirical_rate=0.05,
        samples_evaluated=100
    )
    
    assert profile.rate_ratio() == 5.0  # 0.05 / 0.01


def test_rate_ratio_zero_expected():
    """Rate ratio debe retornar 1.0 si expected_rate es cercano a cero."""
    profile = DetectionRateProfile(
        detector_name='test',
        expected_rate=1e-15,  # Casi cero
        empirical_rate=0.05,
        samples_evaluated=100
    )
    
    assert profile.rate_ratio() == 1.0


def test_calibrated_weights_validation():
    """CalibratedWeights.validate debe verificar suma = 1.0."""
    # Pesos válidos
    valid = CalibratedWeights(
        raw_weights={'a': 0.5, 'b': 0.5},
        calibrated_weights={'a': 0.5, 'b': 0.5},
        calibration_factors={'a': 1.0, 'b': 1.0}
    )
    assert valid.validate()
    
    # Pesos inválidos
    invalid = CalibratedWeights(
        raw_weights={'a': 0.5, 'b': 0.5},
        calibrated_weights={'a': 0.4, 'b': 0.4},  # Suma = 0.8
        calibration_factors={'a': 1.0, 'b': 1.0}
    )
    assert not invalid.validate()


def test_calibration_preserves_detector_names():
    """Calibración debe preservar nombres de detectores."""
    calibrator = EnsembleCalibrator()
    
    raw_weights = {
        'isolation_forest': 0.3,
        'z_score': 0.2,
        'iqr': 0.1,
        'lof': 0.15,
        'velocity_z': 0.15,
        'acceleration_z': 0.1,
    }
    
    profiles = [
        DetectionRateProfile(
            detector_name='isolation_forest',
            expected_rate=0.005,
            empirical_rate=0.05,
            samples_evaluated=1000
        ),
        DetectionRateProfile(
            detector_name='z_score',
            expected_rate=0.005,
            empirical_rate=0.005,
            samples_evaluated=1000
        ),
    ]
    
    result = calibrator.calibrate_by_detection_rate(raw_weights, profiles)
    
    # Todos los nombres deben estar presentes
    for name in raw_weights:
        assert name in result.calibrated_weights
        assert name in result.calibration_factors
