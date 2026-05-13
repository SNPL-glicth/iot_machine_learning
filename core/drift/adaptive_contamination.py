"""Adaptive contamination para IsolationForest con hysteresis.

Principio: SRP - solo maneja adaptación de contamination.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from core.parameters.numerical_constants import STAT_THRESHOLDS


@dataclass
class ContaminationState:
    """Estado de contaminación con historia."""
    current_contamination: float = STAT_THRESHOLDS.CONTAMINATION_DEFAULT
    anomaly_count: int = 0
    total_samples: int = 0
    detection_history: deque[bool] = field(default_factory=lambda: deque(maxlen=100))
    
    @property
    def observed_rate(self) -> float:
        """Tasa de anomalías observada."""
        if self.total_samples == 0:
            return 0.0
        return self.anomaly_count / self.total_samples
    
    def add_detection(self, is_anomaly: bool) -> None:
        """Registra nueva detección."""
        self.detection_history.append(is_anomaly)
        self.total_samples += 1
        if is_anomaly:
            self.anomaly_count += 1


@dataclass
class ContaminationHysteresisConfig:
    """Configuración de hysteresis para contamination."""
    threshold_increase: float = 1.2  # Aumentar si ratio > 1.2
    threshold_decrease: float = 0.8  # Reducir si ratio < 0.8
    adjustment_factor: float = 0.2  # Ajustar 20% por step
    min_samples: int = 50  # Mínimo de muestras para adaptar
    min_contamination: float = STAT_THRESHOLDS.CONTAMINATION_MIN
    max_contamination: float = STAT_THRESHOLDS.CONTAMINATION_MAX
    
    def should_increase(self, ratio: float, samples: int) -> bool:
        """Decide si aumentar contamination."""
        return samples >= self.min_samples and ratio > self.threshold_increase
    
    def should_decrease(self, ratio: float, samples: int) -> bool:
        """Decide si reducir contamination."""
        return samples >= self.min_samples and ratio < self.threshold_decrease


class AdaptiveContamination:
    """Ajusta contamination con hysteresis."""
    
    def __init__(
        self,
        hysteresis_config: Optional[ContaminationHysteresisConfig] = None,
    ):
        self.state = ContaminationState()
        self.hysteresis = hysteresis_config or ContaminationHysteresisConfig()
    
    def add_detection(self, is_anomaly: bool) -> None:
        """Registra nueva detección."""
        self.state.add_detection(is_anomaly)
    
    def update_contamination(self) -> float:
        """Actualiza contamination basándose en tasa observada.
        
        Returns:
            Nuevo valor de contamination.
        """
        observed_rate = self.state.observed_rate
        current = self.state.current_contamination
        
        if current < 1e-9:
            return current
        
        ratio = observed_rate / current
        samples = self.state.total_samples
        
        # Aplicar hysteresis
        if self.hysteresis.should_increase(ratio, samples):
            new_contamination = current * (1.0 + self.hysteresis.adjustment_factor)
        elif self.hysteresis.should_decrease(ratio, samples):
            new_contamination = current * (1.0 - self.hysteresis.adjustment_factor)
        else:
            new_contamination = current
        
        # Clamp a bounds
        new_contamination = max(
            self.hysteresis.min_contamination,
            min(self.hysteresis.max_contamination, new_contamination),
        )
        
        self.state.current_contamination = new_contamination
        return new_contamination
    
    @property
    def current_contamination(self) -> float:
        """Retorna contamination actual."""
        return self.state.current_contamination
    
    def should_refit(self, threshold: float = 0.2) -> bool:
        """Decide si re-entrenar modelo basándose en cambio de contamination.
        
        Args:
            threshold: Umbral de cambio relativo (default 20%).
        
        Returns:
            True si el cambio supera el threshold.
        """
        # Si no hay historia suficiente, no re-entrenar
        if self.state.total_samples < self.hysteresis.min_samples:
            return False
        
        # Calcular cambio relativo desde el valor inicial
        initial = STAT_THRESHOLDS.CONTAMINATION_DEFAULT
        current = self.state.current_contamination
        
        if initial < 1e-9:
            return False
        
        relative_change = abs(current - initial) / initial
        return relative_change > threshold
    
    def reset(self) -> None:
        """Reset estado (útil al detectar drift)."""
        self.state = ContaminationState()
