"""Filtro de mediana con ventana deslizante.

Responsabilidad ÚNICA: suavizado robusto a outliers.
Reemplaza cada valor por la mediana de los últimos ``window_size`` valores.
Agnóstico al dominio.

Ventajas:
- Inmune a spikes aislados (a diferencia de Kalman/EMA).
- Preserva bordes y cambios de nivel.
- Ideal como pre-filtro antes de Kalman para proteger la calibración de R.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Deque, Dict, List, Optional

from iot_machine_learning.infrastructure.ml.interfaces import SignalFilter

logger = logging.getLogger(__name__)


class MedianSignalFilter(SignalFilter):
    """Filtro de mediana con ventana deslizante por serie.

    Mantiene un buffer circular de los últimos ``window_size`` valores
    y retorna la mediana del buffer en cada paso.

    Args:
        window_size: Tamaño de la ventana (debe ser impar y >= 3).
            Ventanas más grandes eliminan más spikes pero añaden lag.
    """

    def __init__(self, window_size: int = 5) -> None:
        if window_size < 3:
            raise ValueError(f"window_size debe ser >= 3, recibido {window_size}")
        if window_size % 2 == 0:
            raise ValueError(
                f"window_size debe ser impar para mediana sin ambigüedad, "
                f"recibido {window_size}"
            )

        self._window_size: int = window_size
        self._buffers: Dict[str, Deque[float]] = {}
        self._lock: threading.Lock = threading.Lock()

    def filter_value(self, series_id: str, value: float) -> float:
        with self._lock:
            buf = self._buffers.get(series_id)

            if buf is None:
                buf = deque(maxlen=self._window_size)
                self._buffers[series_id] = buf

            buf.append(value)
            return _median(buf)

    def filter(
        self,
        values: List[float],
        timestamps: List[float],
    ) -> List[float]:
        if not values:
            return []

        result: List[float] = []
        buf: Deque[float] = deque(maxlen=self._window_size)

        for v in values:
            buf.append(v)
            result.append(_median(buf))

        return result

    def reset(self, series_id: Optional[str] = None) -> None:
        with self._lock:
            if series_id is None:
                self._buffers.clear()
                logger.info("median_reset_all")
            else:
                self._buffers.pop(series_id, None)
                logger.info(
                    "median_reset_series", extra={"series_id": series_id}
                )


def _median(buf: Deque[float]) -> float:
    """Calcula la mediana de un buffer (sin numpy).

    Args:
        buf: Buffer con al menos 1 elemento.

    Returns:
        Mediana del buffer.
    """
    sorted_vals = sorted(buf)
    n = len(sorted_vals)
    mid = n // 2

    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0
