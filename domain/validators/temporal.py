"""Validaciones temporales reutilizables para UTSAE.

Responsabilidad ÚNICA: guards contra timestamps inválidos, orden no
monótono, gaps excesivos y duplicados temporales.
Sin I/O, sin dependencias externas.

Usado por SensorWindow, TimeSeries y cualquier punto de entrada
que reciba datos temporales.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class TemporalValidationError(ValueError):
    """Error de validación temporal para series de tiempo.

    Hereda de ``ValueError`` para compatibilidad con handlers existentes.
    """

    pass


@dataclass(frozen=True)
class TemporalDiagnostic:
    """Resultado de diagnóstico temporal de una ventana.

    Value object puro — sin lógica de negocio.

    Attributes:
        is_monotonic: True si timestamps son estrictamente crecientes.
        n_out_of_order: Cantidad de puntos fuera de orden.
        n_duplicates: Cantidad de timestamps duplicados.
        n_gaps: Cantidad de gaps que exceden max_gap_factor * median_dt.
        gap_indices: Índices donde se detectaron gaps excesivos.
        median_dt: Mediana del intervalo entre puntos consecutivos.
        min_dt: Mínimo intervalo entre puntos consecutivos.
        max_dt: Máximo intervalo entre puntos consecutivos.
    """

    is_monotonic: bool = True
    n_out_of_order: int = 0
    n_duplicates: int = 0
    n_gaps: int = 0
    gap_indices: List[int] = field(default_factory=list)
    median_dt: float = 0.0
    min_dt: float = 0.0
    max_dt: float = 0.0

    @property
    def is_clean(self) -> bool:
        """True si no hay problemas temporales."""
        return (
            self.is_monotonic
            and self.n_duplicates == 0
            and self.n_gaps == 0
        )


def validate_timestamps(
    timestamps: List[float],
    *,
    require_positive: bool = True,
    require_monotonic: bool = False,
) -> None:
    """Valida una lista de timestamps.

    Checks:
    1. No vacía.
    2. Todos los valores son finitos.
    3. (Opcional) Todos positivos (epoch seconds).
    4. (Opcional) Estrictamente monótonos crecientes.

    Args:
        timestamps: Lista de timestamps a validar.
        require_positive: Si True, rechaza timestamps <= 0.
        require_monotonic: Si True, rechaza orden no creciente.

    Raises:
        TemporalValidationError: Si alguna condición falla.
    """
    if not timestamps:
        raise TemporalValidationError("Lista de timestamps vacía")

    for i, ts in enumerate(timestamps):
        if not isinstance(ts, (int, float)):
            raise TemporalValidationError(
                f"Timestamp en posición {i} no es numérico: {type(ts).__name__}"
            )
        if not math.isfinite(ts):
            raise TemporalValidationError(
                f"Timestamp en posición {i} no es finito: {ts}"
            )
        if require_positive and ts <= 0:
            raise TemporalValidationError(
                f"Timestamp en posición {i} no es positivo: {ts}"
            )

    if require_monotonic:
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i - 1]:
                raise TemporalValidationError(
                    f"Timestamps no monótonos en posición {i}: "
                    f"{timestamps[i - 1]} >= {timestamps[i]}"
                )


def diagnose_temporal_quality(
    timestamps: List[float],
    *,
    max_gap_factor: float = 5.0,
) -> TemporalDiagnostic:
    """Diagnostica la calidad temporal de una serie sin rechazarla.

    Útil para logging y métricas sin bloquear el pipeline.

    Args:
        timestamps: Lista de timestamps.
        max_gap_factor: Un gap se considera excesivo si
            dt > max_gap_factor * median_dt.

    Returns:
        ``TemporalDiagnostic`` con métricas de calidad.
    """
    if len(timestamps) < 2:
        return TemporalDiagnostic(
            is_monotonic=True,
            median_dt=0.0,
            min_dt=0.0,
            max_dt=0.0,
        )

    diffs: List[float] = []
    n_out_of_order = 0
    n_duplicates = 0

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        if dt < 0:
            n_out_of_order += 1
        elif dt == 0:
            n_duplicates += 1
        diffs.append(dt)

    is_monotonic = n_out_of_order == 0 and n_duplicates == 0

    # Compute stats only on positive diffs
    positive_diffs = [d for d in diffs if d > 0]
    if not positive_diffs:
        return TemporalDiagnostic(
            is_monotonic=is_monotonic,
            n_out_of_order=n_out_of_order,
            n_duplicates=n_duplicates,
            median_dt=0.0,
            min_dt=0.0,
            max_dt=0.0,
        )

    sorted_diffs = sorted(positive_diffs)
    mid = len(sorted_diffs) // 2
    if len(sorted_diffs) % 2 == 0:
        median_dt = (sorted_diffs[mid - 1] + sorted_diffs[mid]) / 2.0
    else:
        median_dt = sorted_diffs[mid]

    min_dt = sorted_diffs[0]
    max_dt = sorted_diffs[-1]

    # Detect gaps
    gap_threshold = max_gap_factor * median_dt if median_dt > 0 else float("inf")
    gap_indices: List[int] = []
    for i, dt in enumerate(diffs):
        if dt > gap_threshold:
            gap_indices.append(i + 1)  # index of the point after the gap

    return TemporalDiagnostic(
        is_monotonic=is_monotonic,
        n_out_of_order=n_out_of_order,
        n_duplicates=n_duplicates,
        n_gaps=len(gap_indices),
        gap_indices=gap_indices,
        median_dt=median_dt,
        min_dt=min_dt,
        max_dt=max_dt,
    )


def sort_and_deduplicate(
    timestamps: List[float],
    values: List[float],
) -> Tuple[List[float], List[float]]:
    """Ordena por timestamp y elimina duplicados (mantiene último valor).

    Función de reparación — no rechaza datos, los corrige.
    Útil en el boundary adapter antes de construir SensorWindow.

    Args:
        timestamps: Timestamps (pueden estar desordenados).
        values: Valores correspondientes.

    Returns:
        Tupla ``(sorted_timestamps, sorted_values)`` sin duplicados.

    Raises:
        ValueError: Si las listas tienen diferente longitud.
    """
    if len(timestamps) != len(values):
        raise ValueError(
            f"timestamps ({len(timestamps)}) y values ({len(values)}) "
            f"deben tener la misma longitud"
        )

    if not timestamps:
        return [], []

    # Sort by timestamp, stable sort preserves insertion order for equal keys
    paired = sorted(zip(timestamps, values), key=lambda p: p[0])

    # Deduplicate: keep last value for each timestamp
    deduped: dict[float, float] = {}
    for ts, val in paired:
        deduped[ts] = val  # overwrites duplicates with last value

    sorted_ts = list(deduped.keys())
    sorted_vals = [deduped[ts] for ts in sorted_ts]

    return sorted_ts, sorted_vals
