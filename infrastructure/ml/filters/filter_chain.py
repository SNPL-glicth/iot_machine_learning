"""Cadena composable de filtros de señal.

Responsabilidad ÚNICA: aplicar una secuencia de ``SignalFilter`` en orden.
Cada filtro recibe la salida del anterior.

Ejemplo:
    chain = FilterChain([MedianSignalFilter(5), KalmanSignalFilter(Q=1e-5)])
    # Primero elimina spikes con mediana, luego suaviza con Kalman.

Implementa ``SignalFilter`` para ser intercambiable con cualquier filtro
individual.  Agnóstico al dominio.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from iot_machine_learning.infrastructure.ml.interfaces import SignalFilter

logger = logging.getLogger(__name__)


class FilterChain(SignalFilter):
    """Pipeline composable de filtros de señal.

    Aplica filtros en secuencia: output(f1) → input(f2) → ... → output(fN).

    Args:
        filters: Lista ordenada de filtros a aplicar.
            Lista vacía = IdentityFilter (retorna entrada sin modificar).
    """

    def __init__(self, filters: Optional[List[SignalFilter]] = None) -> None:
        self._filters: List[SignalFilter] = list(filters) if filters else []

    @property
    def n_filters(self) -> int:
        """Número de filtros en la cadena."""
        return len(self._filters)

    @property
    def filter_names(self) -> List[str]:
        """Nombres de clase de los filtros en orden."""
        return [type(f).__name__ for f in self._filters]

    def filter_value(self, series_id: str, value: float) -> float:
        result = value
        for f in self._filters:
            result = f.filter_value(series_id, result)
        return result

    def filter(
        self,
        values: List[float],
        timestamps: List[float],
    ) -> List[float]:
        if not values:
            return []

        result = list(values)
        for f in self._filters:
            result = f.filter(result, timestamps)
        return result

    def reset(self, series_id: Optional[str] = None) -> None:
        for f in self._filters:
            f.reset(series_id)
        logger.info(
            "filter_chain_reset",
            extra={
                "series_id": series_id,
                "n_filters": self.n_filters,
                "filters": self.filter_names,
            },
        )
