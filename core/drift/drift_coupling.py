"""Coupling entre drift detection y weight tracking.

Principio: Dependency Inversion - ambos dependen de abstracción.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class DriftEvent:
    """Evento de drift detectado."""
    timestamp: float
    magnitude: float  # Magnitud del drift (0-1)
    detector_name: str  # Qué lo detectó (Page-Hinkley, etc)
    
    @classmethod
    def create_now(cls, magnitude: float, detector: str) -> "DriftEvent":
        """Factory method."""
        return cls(
            timestamp=time.time(),
            magnitude=magnitude,
            detector_name=detector,
        )


class DriftListener(ABC):
    """Interface para componentes que reaccionan a drift."""
    
    @abstractmethod
    def on_drift_detected(self, event: DriftEvent) -> None:
        """Callback cuando se detecta drift."""
        pass


class DriftNotifier:
    """Notificador de eventos de drift (Observer pattern)."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._listeners: list[DriftListener] = []
        return cls._instance
    
    def subscribe(self, listener: DriftListener) -> None:
        """Suscribe un listener."""
        if listener not in self._listeners:
            self._listeners.append(listener)
    
    def unsubscribe(self, listener: DriftListener) -> None:
        """Desuscribe un listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)
    
    def notify(self, event: DriftEvent) -> None:
        """Notifica a todos los listeners."""
        for listener in self._listeners:
            listener.on_drift_detected(event)


class AdaptiveScalerDriftListener(DriftListener):
    """Listener que resetea AdaptiveScaler al detectar drift."""
    
    def __init__(self, scaler: Any):
        self.scaler = scaler
    
    def on_drift_detected(self, event: DriftEvent) -> None:
        """Reset scaler cuando hay drift."""
        if hasattr(self.scaler, "reset"):
            self.scaler.reset()


class WeightTrackerDriftListener(DriftListener):
    """Listener que ajusta BayesianWeightTracker al detectar drift."""
    
    def __init__(self, tracker: Any):
        self.tracker = tracker
        self.drift_decay_factor = 0.5
        self.drift_variance_expansion = 2.0
    
    def on_drift_detected(self, event: DriftEvent) -> None:
        """Ajusta prior y varianza del tracker."""
        if hasattr(self.tracker, "prior_mean"):
            self.tracker.prior_mean *= self.drift_decay_factor
        
        if hasattr(self.tracker, "prior_variance"):
            self.tracker.prior_variance *= self.drift_variance_expansion
