"""Adapter de compatibilidad para SlidingWindowStore legacy.

DEPRECADO. Delegando a SlidingWindowCache canónico.
Migrar a: infrastructure.persistence.sliding_window.SlidingWindowCache
"""

from __future__ import annotations

import warnings
from typing import List, Optional

from iot_machine_learning.domain.entities.iot.sensor_reading import Reading
from iot_machine_learning.infrastructure.persistence.sliding_window import (
    SlidingWindowCache,
)


class SlidingWindowStore(SlidingWindowCache[Reading]):
    """DEPRECADO. Usa SlidingWindowCache directamente.
    
    Este adapter mantiene compatibilidad con código legacy que usa:
    - ml_service/consumers/sliding_window.SlidingWindowStore
    
    Diferencias con el original:
    - No tiene flush_callback (se puede agregar si es necesario)
    - No tiene cleanup thread proactivo (se puede agregar si es necesario)
    - No tiene max_total_entries (solo max_series)
    """
    
    def __init__(
        self,
        max_size: int = 20,
        max_sensors: int = 1000,
        ttl_seconds: float = 3600.0,
        enable_proactive_cleanup: bool = True,
        max_total_entries: int = 50000,
        flush_callback=None,
    ) -> None:
        """Inicializa adapter con parámetros legacy."""
        warnings.warn(
            "SlidingWindowStore está deprecado. "
            "Usa infrastructure.persistence.sliding_window.SlidingWindowCache",
            DeprecationWarning,
            stacklevel=2,
        )
        
        # Ignorar parámetros no soportados
        if flush_callback is not None:
            warnings.warn(
                "flush_callback no soportado en SlidingWindowCache",
                UserWarning,
            )
        
        # Delegar a canónico
        super().__init__(
            window_size=max_size,
            max_series=max_sensors,
            ttl_seconds=int(ttl_seconds),
        )
    
    def append(self, reading: Reading) -> int:
        """Agrega Reading a la ventana (compatibilidad legacy)."""
        return super().append(
            series_id=reading.series_id,
            value=reading,
            timestamp=reading.timestamp,
        )
    
    def get_window(self, series_id: str) -> List[Reading]:
        """Obtiene ventana como lista de Reading (compatibilidad legacy)."""
        items = self.get_values(series_id)
        if items is None:
            return []
        return items
    
    def close(self) -> None:
        """No-op para compatibilidad legacy."""
        pass
