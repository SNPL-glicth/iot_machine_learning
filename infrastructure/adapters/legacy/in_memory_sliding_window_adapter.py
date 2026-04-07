"""Adapter de compatibilidad para InMemorySlidingWindowStore legacy.

DEPRECADO. Delegando a SlidingWindowCache canónico.
Migrar a: infrastructure.persistence.sliding_window.SlidingWindowCache
"""

from __future__ import annotations

import warnings
from typing import Generic, List, TypeVar

from iot_machine_learning.infrastructure.persistence.sliding_window import (
    SlidingWindowCache,
)

T = TypeVar("T")


class InMemorySlidingWindowStore(SlidingWindowCache[T], Generic[T]):
    """DEPRECADO. Usa SlidingWindowCache directamente.
    
    Este adapter mantiene compatibilidad con código legacy que usa:
    - infrastructure/sliding_window/in_memory.InMemorySlidingWindowStore
    
    Diferencias con el original:
    - Usa series_id: str en lugar de sensor_id: int
    - No usa WindowConfig (parámetros directos)
    """
    
    def __init__(self, config=None) -> None:
        """Inicializa adapter con config legacy."""
        warnings.warn(
            "InMemorySlidingWindowStore está deprecado. "
            "Usa infrastructure.persistence.sliding_window.SlidingWindowCache",
            DeprecationWarning,
            stacklevel=2,
        )
        
        # Extract params from config
        if config is not None:
            max_size = getattr(config, 'max_size', 20)
            max_sensors = getattr(config, 'max_sensors', 1000)
            ttl_seconds = getattr(config, 'ttl_seconds', 3600)
        else:
            max_size = 20
            max_sensors = 1000
            ttl_seconds = 3600
        
        # Delegar a canónico
        super().__init__(
            window_size=max_size,
            max_series=max_sensors,
            ttl_seconds=int(ttl_seconds),
        )
    
    def append(self, sensor_id: int, item: T, timestamp: float) -> int:
        """Agrega item (compatibilidad con sensor_id: int)."""
        return super().append(str(sensor_id), item, timestamp)
    
    def get_window(self, sensor_id: int) -> List[T]:
        """Obtiene ventana (compatibilidad con sensor_id: int)."""
        items = self.get_values(str(sensor_id))
        if items is None:
            return []
        return items
    
    def get_size(self, sensor_id: int) -> int:
        """Tamaño de ventana (compatibilidad con sensor_id: int)."""
        return self.size(str(sensor_id))
    
    def clear(self, sensor_id: int) -> None:
        """Limpia ventana (compatibilidad con sensor_id: int)."""
        super().clear(str(sensor_id))
    
    def sensor_ids(self) -> List[int]:
        """Lista de sensor_ids como int (compatibilidad legacy)."""
        str_ids = self.series_ids()
        # Intentar convertir a int, ignorar los que no se puedan
        int_ids = []
        for sid in str_ids:
            try:
                int_ids.append(int(sid))
            except ValueError:
                pass
        return int_ids
    
    def evict_stale(self, current_time: float) -> int:
        """Evict stale entries (no-op en canónico, TTL automático)."""
        return 0
