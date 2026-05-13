"""ADWIN drift detector — adaptive windowing for concept drift.

Simplified implementation of Adaptive Windowing (ADWIN) algorithm.
Maintains adaptive window that shrinks when drift is detected.

CUÁNDO USAR ADWIN vs Page-Hinkley (FASE-23):
- **ADWIN:** Drift abrupto o heteroscedástico (varianza cambiante).
  Mejor para: cambios de régimen bruscos, datos no estacionarios.
  Limitación: mayor costo computacional (O(log n) vs O(1)).
- **Page-Hinkley:** Drift gradual y unidireccional (mean shift).
  Mejor para: tendencias, degradación lenta de sensores.
  Limitación: asume cambios monotónicos, lento en drift abrupto.
Default: Page-Hinkley (más eficiente, cubre mayoría de casos IoT).
Ver también: infrastructure/ml/cognitive/drift/page_hinkley.py

BACKLOG (Low Priority - FASE-24):
- Hybrid drift detection: combinar ADWIN + Page-Hinkley con voting
- Gradual window reset post-drift (actualmente shrink_factor=0.5 abrupto)
- Automatic detector selection por características de datos
Ver: page_hinkley.py para backlog simétrico

Reference:
    Bifet & Gavaldà (2007). "Learning from Time-Changing Data with
    Adaptive Windowing". SIAM International Conference on Data Mining.
"""

from __future__ import annotations

import logging
import math
from collections import deque
from typing import Deque, Optional

logger = logging.getLogger(__name__)


class ADWINDetector:
    """ADWIN drift detector with adaptive windowing.
    
    Detects drift by comparing statistics of two sub-windows.
    Window shrinks automatically when drift is detected.
    
    Thread-safe: no shared mutable state across instances.
    
    Attributes:
        _delta: Confidence parameter (smaller = more sensitive).
        _window: Sliding window of recent observations.
        _max_window_size: Maximum window capacity.
        _total: Sum of values in window.
        _variance: Variance estimate.
    """
    
    def __init__(
        self,
        delta: float = 0.002,
        max_window_size: int = 1000,
    ) -> None:
        """Initialize ADWIN detector.
        
        Args:
            delta: Confidence parameter (0 < delta < 1).
                PENDING_CALIBRATION: Equivale a confidence = 1 - delta = 99.8%.
                delta=0.1 para menos sensibilidad, delta=0.001 para más sensibilidad.
                Ackermann et al. recomiendan 0.002 como default robusto.
            
            max_window_size: Maximum window size.
                A 1Hz = 1000 segundos (~16 min) de memoria máxima.
                A 10Hz = 100 segundos de memoria máxima.
                PENDING_CALIBRATION: Ajustar como frecuencia_hz * segundos_deseados.
                
            shrink_factor: (hardcoded=0.5) Reduce ventana 50% al detectar drift.
                PENDING_CALIBRATION: Considerar 0.75 para reset más gradual.
                Ver backlog: gradual reset post-drift.
        """
        if not 0 < delta < 1:
            raise ValueError(f"delta must be in (0, 1), got {delta}")
        if max_window_size < 10:
            raise ValueError(f"max_window_size must be >= 10, got {max_window_size}")
        
        self._delta = delta
        self._max_window_size = max_window_size
        self._window: Deque[float] = deque(maxlen=max_window_size)
        self._total = 0.0
        self._variance = 0.0
    
    def update(self, value: float) -> bool:
        """Process new observation and check for drift.
        
        Args:
            value: New observation value.
        
        Returns:
            True if drift detected, False otherwise.
        """
        # Add new value to window
        if len(self._window) == self._max_window_size:
            # Remove oldest value from total
            self._total -= self._window[0]
        
        self._window.append(value)
        self._total += value
        
        # Need at least 2 observations
        if len(self._window) < 2:
            return False
        
        # Update variance estimate
        mean = self._total / len(self._window)
        self._variance = sum((x - mean) ** 2 for x in self._window) / len(self._window)
        
        # Check for drift using Hoeffding bound
        drift_detected = self._detect_change()
        
        if drift_detected:
            logger.debug(
                "adwin_drift_detected",
                extra={
                    "window_size": len(self._window),
                    "mean": round(mean, 4),
                    "variance": round(self._variance, 4),
                },
            )
            # Shrink window by half
            n_remove = len(self._window) // 2
            for _ in range(n_remove):
                removed = self._window.popleft()
                self._total -= removed
        
        return drift_detected
    
    def _detect_change(self) -> bool:
        """Detect change using Hoeffding bound on sub-windows."""
        n = len(self._window)
        if n < 10:
            return False
        
        # Split window in half
        n0 = n // 2
        n1 = n - n0
        
        # Compute means of sub-windows
        mean0 = sum(list(self._window)[:n0]) / n0
        mean1 = sum(list(self._window)[n0:]) / n1
        
        # Hoeffding bound
        m = 1.0 / (1.0 / n0 + 1.0 / n1)
        epsilon = math.sqrt((1.0 / (2.0 * m)) * math.log(4.0 / self._delta))
        
        # Drift if difference exceeds bound
        return abs(mean0 - mean1) > epsilon
    
    def reset(self) -> None:
        """Reset detector state."""
        self._window.clear()
        self._total = 0.0
        self._variance = 0.0
    
    @property
    def window_size(self) -> int:
        """Current window size."""
        return len(self._window)
    
    @property
    def mean(self) -> float:
        """Current window mean."""
        if len(self._window) == 0:
            return 0.0
        return self._total / len(self._window)
