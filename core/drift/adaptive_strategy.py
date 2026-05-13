"""Estrategia unificada de adaptación con hysteresis.

Principio: Open/Closed - extensible sin modificar.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np

from core.parameters.numerical_constants import EPSILON


@dataclass
class AdaptiveState:
    """Estado de adaptación con hysteresis."""
    current_scale: float = 1.0
    previous_scale: float = 1.0
    volatility_history: Deque[float] = field(default_factory=lambda: deque(maxlen=20))
    is_in_transition: bool = False
    
    def add_volatility(self, value: float) -> None:
        """Agrega nueva medida de volatilidad."""
        self.volatility_history.append(value)
    
    def mean_volatility(self) -> float:
        """Promedio de volatilidad reciente."""
        if not self.volatility_history:
            return 1.0
        return float(np.mean(self.volatility_history))


@dataclass
class HysteresisConfig:
    """Configuración de hysteresis para prevenir oscilación."""
    threshold_increase: float = 1.2  # Aumentar scale si ratio > 1.2
    threshold_decrease: float = 0.8  # Reducir scale si ratio < 0.8
    smooth_factor: float = 0.3  # EMA smoothing (30% new, 70% old)
    min_samples: int = 5  # Mínimo para decisión
    
    def should_increase(self, ratio: float, samples: int) -> bool:
        """Decide si aumentar scale."""
        return samples >= self.min_samples and ratio > self.threshold_increase
    
    def should_decrease(self, ratio: float, samples: int) -> bool:
        """Decide si reducir scale."""
        return samples >= self.min_samples and ratio < self.threshold_decrease


class AdaptiveScaler:
    """Scaler adaptativo con hysteresis para prevenir oscilación.
    
    Estrategia:
    1. Smoothing de volatilidad con EMA
    2. Hysteresis bands (0.8-1.2) para cambios de scale
    3. Transición gradual entre scales
    """
    
    def __init__(
        self,
        scale_min: float = 0.5,
        scale_max: float = 5.0,
        hysteresis_config: HysteresisConfig | None = None,
    ):
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.hysteresis = hysteresis_config or HysteresisConfig()
        self.state = AdaptiveState()
    
    def compute_scale(
        self,
        current_volatility: float,
        base_volatility: float,
    ) -> float:
        """Computa scale factor con hysteresis.
        
        Args:
            current_volatility: Volatilidad actual (rolling std)
            base_volatility: Volatilidad base (initial std)
            
        Returns:
            Scale factor en [scale_min, scale_max]
        """
        self.state.add_volatility(current_volatility)
        
        if base_volatility < EPSILON.DIVISION:
            return 1.0
        
        raw_ratio = current_volatility / base_volatility
        smoothed_ratio = self._smooth_ratio(raw_ratio)
        new_scale = self._apply_hysteresis(smoothed_ratio)
        new_scale = max(self.scale_min, min(self.scale_max, new_scale))
        final_scale = self._gradual_transition(new_scale)
        
        self.state.previous_scale = self.state.current_scale
        self.state.current_scale = final_scale
        
        return final_scale
    
    def _smooth_ratio(self, raw_ratio: float) -> float:
        """Smoothing de ratio con EMA."""
        if not self.state.volatility_history:
            return raw_ratio
        
        mean_vol = self.state.mean_volatility()
        if mean_vol < EPSILON.DIVISION:
            return raw_ratio
        
        alpha = self.hysteresis.smooth_factor
        smoothed = alpha * raw_ratio + (1 - alpha) * self.state.current_scale
        return smoothed
    
    def _apply_hysteresis(self, ratio: float) -> float:
        """Aplica hysteresis bands."""
        samples = len(self.state.volatility_history)
        current = self.state.current_scale
        
        if not (
            self.hysteresis.should_increase(ratio, samples) or
            self.hysteresis.should_decrease(ratio, samples)
        ):
            return current
        
        return ratio
    
    def _gradual_transition(self, target_scale: float) -> float:
        """Transición gradual para evitar saltos bruscos."""
        current = self.state.current_scale
        max_change = 0.5
        
        delta = target_scale - current
        if abs(delta) > max_change:
            direction = 1 if delta > 0 else -1
            return current + (direction * max_change)
        
        return target_scale
    
    def reset(self) -> None:
        """Reset estado (útil al detectar drift)."""
        self.state = AdaptiveState()


class UnifiedAdaptiveConfig:
    """Configuración unificada para todos los detectores."""
    
    ADAPTIVE_ENABLED: bool = True  # Todos adaptativos
    MAX_HISTORY: int = 100
    MIN_HISTORY_ENTRIES: int = 5
    SCALE_MIN: float = 0.5
    SCALE_MAX: float = 5.0
    HYSTERESIS_INCREASE: float = 1.2
    HYSTERESIS_DECREASE: float = 0.8
    SMOOTH_FACTOR: float = 0.3
