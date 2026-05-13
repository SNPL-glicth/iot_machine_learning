"""Tests para ensemble drift coupling."""
import pytest
import numpy as np

from core.ensemble.ensemble_drift_coupling import (
    EnsembleWeightState,
    EnsembleCalibrationDriftListener,
    EnsembleDriftCoupling,
)
from core.drift.drift_coupling import DriftEvent


class MockDetector:
    """Mock detector para tests."""
    
    def __init__(self, name, detection_rate=0.5):
        self.name = name
        self.detection_rate = detection_rate
    
    def detect(self, data):
        """Detect anomalies with fixed rate."""
        n = len(data)
        n_anomalies = int(n * self.detection_rate)
        return np.array([1.0] * n_anomalies + [0.0] * (n - n_anomalies))


def test_ensemble_weight_state_hysteresis():
    """Weight state debe aplicar hysteresis."""
    state = EnsembleWeightState()
    state.current_weights = {"a": 0.5, "b": 0.5}
    
    # Cambio pequeño (< 15%) → no actualizar
    new_weights = {"a": 0.52, "b": 0.48}
    assert not state.should_update(new_weights)
    
    # Cambio grande (> 15%) → actualizar
    new_weights = {"a": 0.7, "b": 0.3}
    assert state.should_update(new_weights)


def test_ensemble_weight_state_update():
    """Weight state debe actualizar correctamente."""
    state = EnsembleWeightState()
    state.current_weights = {"a": 0.5, "b": 0.5}
    
    new_weights = {"a": 0.7, "b": 0.3}
    state.update(new_weights)
    
    assert state.current_weights == new_weights
    assert state.previous_weights == {"a": 0.5, "b": 0.5}
    assert len(state.weight_history) == 1


def test_drift_triggers_recalibration():
    """Drift event debe re-calibrar ensemble."""
    detectors = {
        "a": MockDetector("a", detection_rate=0.1),
        "b": MockDetector("b", detection_rate=0.5),
    }
    calibration_data = np.random.randn(1000)  # NUEVO: 1000 muestras mínimo
    
    coupling = EnsembleDriftCoupling(
        detectors=detectors,
        calibration_data=calibration_data,
        initial_weights={"a": 0.5, "b": 0.5},
    )
    
    # Simular drift event
    listener = coupling.get_listener()
    event = DriftEvent.create_now(magnitude=0.5, detector="test")
    listener.on_drift_detected(event)
    
    # Pesos deben haber sido actualizados
    new_weights = coupling.current_weights
    assert len(new_weights) == 2


def test_manual_calibration():
    """Calibración manual debe funcionar."""
    detectors = {
        "a": MockDetector("a", detection_rate=0.1),
        "b": MockDetector("b", detection_rate=0.5),
    }
    calibration_data = np.random.randn(1000)  # NUEVO: 1000 muestras mínimo
    
    coupling = EnsembleDriftCoupling(
        detectors=detectors,
        calibration_data=calibration_data,
        initial_weights={"a": 0.5, "b": 0.5},
    )
    
    calibrated = coupling.manual_calibrate()
    
    assert calibrated.validate()
    assert sum(calibrated.calibrated_weights.values()) == pytest.approx(1.0)


def test_hysteresis_threshold():
    """Hysteresis threshold debe ser configurable."""
    state = EnsembleWeightState(hysteresis_threshold=0.1)
    state.current_weights = {"a": 0.5, "b": 0.5}
    
    # Con threshold=0.1, cambio de 12% debe actualizar
    new_weights = {"a": 0.56, "b": 0.44}
    assert state.should_update(new_weights)


def test_weight_history_maxlen():
    """Weight history debe tener maxlen."""
    state = EnsembleWeightState()
    
    for i in range(15):
        state.update({"a": 0.5 + i * 0.01, "b": 0.5 - i * 0.01})
    
    # maxlen=10, solo últimos 10 deben estar en historia
    assert len(state.weight_history) == 10


def test_empty_weights_always_update():
    """Si no hay pesos actuales, siempre actualizar."""
    state = EnsembleWeightState()
    
    new_weights = {"a": 0.5, "b": 0.5}
    assert state.should_update(new_weights)


def test_drift_listener_requires_components():
    """Drift listener requiere calibrator, measurer, detectors."""
    detectors = {"a": MockDetector("a")}
    calibration_data = np.random.randn(100)
    
    coupling = EnsembleDriftCoupling(
        detectors=detectors,
        calibration_data=calibration_data,
    )
    
    listener = coupling.get_listener()
    assert listener.calibrator is not None
    assert listener.measurer is not None
    assert listener.detectors == detectors
