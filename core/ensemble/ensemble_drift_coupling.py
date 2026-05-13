"""Coupling entre drift detection y ensemble calibration.

Principio: Observer pattern - drift events trigger re-calibration.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from core.drift.drift_coupling import DriftEvent, DriftListener
from core.ensemble.ensemble_calibrator import (
    CalibratedWeights,
    DetectionRateProfile,
    EnsembleCalibrator,
    DetectionRateMeasurer,
)


@dataclass
class EnsembleWeightState:
    """Estado de pesos con hysteresis."""
    current_weights: Dict[str, float] = field(default_factory=dict)
    previous_weights: Dict[str, float] = field(default_factory=dict)
    weight_history: deque[Dict[str, float]] = field(
        default_factory=lambda: deque(maxlen=10)
    )
    last_calibration_time: Optional[float] = None
    hysteresis_threshold: float = 0.15  # 15% cambio relativo mínimo
    
    def should_update(self, new_weights: Dict[str, float]) -> bool:
        """Decide si actualizar pesos basándose en hysteresis."""
        if not self.current_weights:
            return True
        
        # Calcular cambio relativo promedio
        total_change = 0.0
        count = 0
        for name, new_weight in new_weights.items():
            if name in self.current_weights:
                old_weight = self.current_weights[name]
                if old_weight > 0:
                    relative_change = abs(new_weight - old_weight) / old_weight
                    total_change += relative_change
                    count += 1
        
        if count == 0:
            return True
        
        avg_change = total_change / count
        return avg_change > self.hysteresis_threshold
    
    def update(self, new_weights: Dict[str, float]) -> None:
        """Actualiza estado con nuevos pesos."""
        self.previous_weights = dict(self.current_weights)
        self.current_weights = dict(new_weights)
        self.weight_history.append(dict(new_weights))
        
        import time
        self.last_calibration_time = time.time()


class EnsembleCalibrationDriftListener(DriftListener):
    """Listener que re-calibra ensemble al detectar drift."""
    
    def __init__(
        self,
        calibrator: EnsembleCalibrator,
        measurer: DetectionRateMeasurer,
        detectors: Dict[str, object],
        calibration_data,
        weight_state: EnsembleWeightState,
    ):
        self.calibrator = calibrator
        self.measurer = measurer
        self.detectors = detectors
        self.calibration_data = calibration_data
        self.weight_state = weight_state
    
    def on_drift_detected(self, event: DriftEvent) -> None:
        """Re-calibra pesos cuando se detecta drift."""
        # Medir tasas de detección actuales
        profiles = self.measurer.measure_rates(
            self.detectors,
            self.calibration_data,
        )
        
        # Calibrar pesos con tasas actuales
        raw_weights = self.weight_state.current_weights or {
            name: 1.0 / len(self.detectors)
            for name in self.detectors.keys()
        }
        
        calibrated = self.calibrator.calibrate_by_detection_rate(
            raw_weights,
            profiles,
        )
        
        new_weights = calibrated.calibrated_weights
        
        # Aplicar hysteresis
        if self.weight_state.should_update(new_weights):
            self.weight_state.update(new_weights)
        else:
            # Mantener pesos actuales (cambio muy pequeño)
            pass


class EnsembleDriftCoupling:
    """Acopla drift detection a ensemble calibration con hysteresis."""
    
    def __init__(
        self,
        detectors: Dict[str, object],
        calibration_data,
        initial_weights: Optional[Dict[str, float]] = None,
    ):
        self.detectors = detectors
        self.calibration_data = calibration_data
        self.calibrator = EnsembleCalibrator()
        self.measurer = DetectionRateMeasurer()
        self.weight_state = EnsembleWeightState()
        
        if initial_weights:
            self.weight_state.current_weights = dict(initial_weights)
    
    def get_listener(self) -> EnsembleCalibrationDriftListener:
        """Retorna listener para suscribir a drift events."""
        return EnsembleCalibrationDriftListener(
            self.calibrator,
            self.measurer,
            self.detectors,
            self.calibration_data,
            self.weight_state,
        )
    
    @property
    def current_weights(self) -> Dict[str, float]:
        """Retorna pesos actuales."""
        return self.weight_state.current_weights
    
    def manual_calibrate(self) -> CalibratedWeights:
        """Calibración manual (sin drift event)."""
        profiles = self.measurer.measure_rates(
            self.detectors,
            self.calibration_data,
        )
        
        raw_weights = self.weight_state.current_weights or {
            name: 1.0 / len(self.detectors)
            for name in self.detectors.keys()
        }
        
        calibrated = self.calibrator.calibrate_by_detection_rate(
            raw_weights,
            profiles,
        )
        
        self.weight_state.update(calibrated.calibrated_weights)
        return calibrated
