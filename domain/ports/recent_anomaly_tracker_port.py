"""RecentAnomalyTrackerPort — seguimiento de anomalías recientes por serie.

Port para mantener ventana deslizante de anomalías recientes (2 horas).
Usado por ContextualDecisionEngine para calcular scores contextuales.

Backend: Redis SortedSet o memoria (configurable vía ML_ANOMALY_TRACKER_BACKEND).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)


class RecentAnomalyTrackerPort(ABC):
    """Port interface para seguimiento de anomalías recientes.
    
    Responsabilidad: mantener ventana deslizante de anomalías por serie,
    permitiendo calcular tasas y conteos para decisiones contextuales.
    """
    
    @abstractmethod
    def record_anomaly(
        self,
        series_id: str,
        anomaly_score: float,
        timestamp: Optional[float] = None,
        regime: str = "",
    ) -> None:
        """Registrar una anomalía detectada.
        
        Args:
            series_id: Identificador de la serie
            anomaly_score: Score de anomalía [0, 1]
            timestamp: Unix timestamp (default: ahora)
            regime: Régimen actual opcional
        """
        ...
    
    @abstractmethod
    def record_normal(
        self,
        series_id: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """Registrar una predicción normal (sin anomalía).
        
        Necesario para calcular ratio anomalías/predicciones totales
        y resetear contador de anomalías consecutivas.
        
        Args:
            series_id: Identificador de la serie
            timestamp: Unix timestamp (default: ahora)
        """
        ...
    
    @abstractmethod
    def get_count_last_n_minutes(self, series_id: str, minutes: int) -> int:
        """Contar anomalías en ventana temporal.
        
        Args:
            series_id: Identificador de la serie
            minutes: Ventana temporal en minutos
        
        Returns:
            Número de anomalías en la ventana
        """
        ...
    
    @abstractmethod
    def get_consecutive_count(self, series_id: str) -> int:
        """Contar anomalías consecutivas sin interrupción.
        
        Se resetea cuando hay una predicción normal.
        
        Returns:
            Número de anomalías consecutivas
        """
        ...
    
    @abstractmethod
    def get_anomaly_rate(self, series_id: str, window_minutes: int) -> float:
        """Calcular ratio anomalías / predicciones totales.
        
        Args:
            series_id: Identificador de la serie
            window_minutes: Ventana temporal en minutos
        
        Returns:
            Ratio [0.0, 1.0] (0.0 si no hay predicciones)
        """
        ...
    
    @abstractmethod
    def reset(self, series_id: Optional[str] = None) -> None:
        """Resetear historial.
        
        Args:
            series_id: Si None, resetear todas las series
        """
        ...


class NullAnomalyTracker(RecentAnomalyTrackerPort):
    """No-op implementation — devuelve ceros en todos los métodos.
    
    Usado cuando no hay tracker configurado (backward compat).
    """
    
    def record_anomaly(
        self,
        series_id: str,
        anomaly_score: float,
        timestamp: Optional[float] = None,
        regime: str = "",
    ) -> None:
        """No-op."""
        pass
    
    def record_normal(
        self,
        series_id: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """No-op."""
        pass
    
    def get_count_last_n_minutes(self, series_id: str, minutes: int) -> int:
        """Siempre devuelve 0."""
        return 0
    
    def get_consecutive_count(self, series_id: str) -> int:
        """Siempre devuelve 0."""
        return 0
    
    def get_anomaly_rate(self, series_id: str, window_minutes: int) -> float:
        """Siempre devuelve 0.0."""
        return 0.0
    
    def reset(self, series_id: Optional[str] = None) -> None:
        """No-op."""
        pass
