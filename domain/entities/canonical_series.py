"""Tipos canónicos para series temporales — agnósticos al dominio.

Reemplaza el modelo IoT legacy (sensor_id:int) con el modelo Zenin (series_id:UUID).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import UUID


@dataclass(frozen=True)
class DataPoint:
    """Punto de dato de una serie temporal.
    
    Inmutable. Representa una observación única en el tiempo.
    
    Attributes:
        value: Valor numérico de la observación
        timestamp: Timestamp Unix (segundos desde epoch)
        quality: Factor de calidad [0-1], 1.0 = perfecto
        metadata: Metadatos adicionales (ej: sensor_id legacy)
    """
    
    value: float
    timestamp: float
    quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimeWindow:
    """Ventana temporal de una serie. Inmutable.
    
    Representa un conjunto ordenado de observaciones de una serie
    en un intervalo de tiempo.
    
    Attributes:
        series_id: Identificador UUID de la serie (Zenin canonical)
        tenant_id: Identificador UUID del tenant (multi-tenancy)
        points: Tupla inmutable de DataPoints ordenados cronológicamente
    """
    
    series_id: UUID
    tenant_id: UUID
    points: tuple  # tuple[DataPoint, ...]
    
    @property
    def values(self) -> List[float]:
        """Lista de valores en orden cronológico."""
        return [p.value for p in self.points]
    
    @property
    def timestamps(self) -> List[float]:
        """Lista de timestamps en orden cronológico."""
        return [p.timestamp for p in self.points]
    
    @property
    def size(self) -> int:
        """Número de puntos en la ventana."""
        return len(self.points)
    
    @property
    def is_empty(self) -> bool:
        """Verdadero si la ventana no tiene puntos."""
        return len(self.points) == 0
    
    def latest(self) -> Optional[DataPoint]:
        """Último punto de la ventana, o None si está vacía."""
        return self.points[-1] if self.points else None
    
    def earliest(self) -> Optional[DataPoint]:
        """Primer punto de la ventana, o None si está vacía."""
        return self.points[0] if self.points else None
    
    @property
    def duration_seconds(self) -> float:
        """Duración de la ventana en segundos."""
        if len(self.points) < 2:
            return 0.0
        return self.points[-1].timestamp - self.points[0].timestamp
