from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import Deque, Dict, Iterable, Tuple


@dataclass(frozen=True)
class WindowStats:
    """Estadísticos agregados para una ventana de tiempo.

    Se calculan sobre lecturas recientes dentro de [now - window, now].

    Además de media/mínimo/máximo y tendencia, expone la desviación estándar
    simple de la ventana (`std_dev`) para poder calcular tolerancias dinámicas
    (p.ej. z-score) y el último valor observado (`last_value`).
    """

    window_seconds: float
    mean: float
    min: float
    max: float
    std_dev: float
    trend: float  # pendiente aproximada valor/segundo
    count: int
    last_value: float


class SlidingWindowBuffer:
    """Buffer deslizante en memoria por sensor.

    - Mantiene un buffer por sensor con las últimas lecturas hasta
      `max_horizon_seconds`.
    - Permite calcular ventanas deslizantes (p.ej. 1s, 5s, 10s) con
      agregados simples: avg, min, max, tendencia.
    """

    def __init__(self, max_horizon_seconds: float = 10.0) -> None:
        self._max_horizon_seconds = float(max_horizon_seconds)
        # sensor_id -> deque[(timestamp, value)]
        self._buffers: Dict[int, Deque[Tuple[float, float]]] = {}

    def add_reading(
        self,
        sensor_id: int,
        value: float,
        timestamp: float,
        windows: Iterable[float] = (1.0, 5.0, 10.0),
    ) -> Dict[str, WindowStats]:
        """Añade una lectura y devuelve stats por ventana.

        Devuelve un dict tipo {"w1": WindowStats, "w5": WindowStats, ...}.
        Solo se incluyen ventanas que tengan al menos 1 punto.
        """

        buf = self._buffers.setdefault(sensor_id, deque())
        buf.append((timestamp, float(value)))

        # Recortar lecturas fuera del horizonte máximo
        cutoff = timestamp - self._max_horizon_seconds
        while buf and buf[0][0] < cutoff:
            buf.popleft()

        return self._compute_stats(buf, timestamp, windows)

    def _compute_stats(
        self,
        buf: Deque[Tuple[float, float]],
        now_ts: float,
        windows: Iterable[float],
    ) -> Dict[str, WindowStats]:
        result: Dict[str, WindowStats] = {}
        if not buf:
            return result

        for w in windows:
            w = float(w)
            cutoff = now_ts - w
            window_points = [item for item in buf if item[0] >= cutoff]
            if not window_points:
                continue

            values = [v for (_, v) in window_points]
            v_min = min(values)
            v_max = max(values)
            v_mean = sum(values) / len(values)

            # Desviación estándar poblacional simple (sin Bessel) suficiente
            # para tolerancias dinámicas en ventanas cortas.
            if len(values) >= 2:
                mean_diff_sq = [(v - v_mean) ** 2 for v in values]
                std_dev = sqrt(sum(mean_diff_sq) / len(values))
            else:
                std_dev = 0.0

            if len(window_points) >= 2:
                t0, v0 = window_points[0]
                t1, v1 = window_points[-1]
                dt = max(t1 - t0, 1e-3)  # evitar división por 0
                trend = (v1 - v0) / dt
            else:
                trend = 0.0

            key = f"w{int(w)}"
            result[key] = WindowStats(
                window_seconds=w,
                mean=v_mean,
                min=v_min,
                max=v_max,
                std_dev=std_dev,
                trend=trend,
                count=len(values),
                last_value=values[-1],
            )

        return result
