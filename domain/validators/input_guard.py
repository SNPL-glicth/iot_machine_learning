"""Guards de entrada para use cases UTSAE.

Responsabilidad ÚNICA: validar inputs en el boundary de los use cases
antes de que lleguen al dominio. Rechaza datos corruptos temprano.
Sin I/O, sin dependencias externas.

Checks:
- sensor_id / series_id presente y válido
- window_size dentro de rangos razonables
- timestamps no en el futuro
- valores numéricos finitos
"""

from __future__ import annotations

import math
import time
from typing import List, Optional


class InputGuardError(ValueError):
    """Error de validación de input en boundary de use case.

    Hereda de ``ValueError`` para compatibilidad con handlers existentes.
    """

    pass


def guard_sensor_id(sensor_id: object) -> int:
    """Valida y convierte sensor_id a int.

    Args:
        sensor_id: Valor a validar.

    Returns:
        sensor_id como int positivo.

    Raises:
        InputGuardError: Si inválido.
    """
    if sensor_id is None:
        raise InputGuardError("sensor_id es requerido")

    try:
        sid = int(sensor_id)
    except (TypeError, ValueError):
        raise InputGuardError(
            f"sensor_id debe ser entero, recibido {type(sensor_id).__name__}"
        )

    if sid <= 0:
        raise InputGuardError(f"sensor_id debe ser positivo, recibido {sid}")

    return sid


def guard_series_id(series_id: object) -> str:
    """Valida series_id como string no vacío.

    Args:
        series_id: Valor a validar.

    Returns:
        series_id como str.

    Raises:
        InputGuardError: Si inválido.
    """
    if series_id is None:
        raise InputGuardError("series_id es requerido")

    sid = str(series_id).strip()
    if not sid:
        raise InputGuardError("series_id no puede estar vacío")

    return sid


def guard_window_size(
    window_size: int,
    *,
    min_size: int = 1,
    max_size: int = 10_000,
) -> int:
    """Valida window_size dentro de rangos razonables.

    Args:
        window_size: Tamaño de ventana solicitado.
        min_size: Mínimo permitido.
        max_size: Máximo permitido.

    Returns:
        window_size validado.

    Raises:
        InputGuardError: Si fuera de rango.
    """
    if not isinstance(window_size, int):
        raise InputGuardError(
            f"window_size debe ser entero, recibido {type(window_size).__name__}"
        )

    if window_size < min_size:
        raise InputGuardError(
            f"window_size debe ser >= {min_size}, recibido {window_size}"
        )

    if window_size > max_size:
        raise InputGuardError(
            f"window_size debe ser <= {max_size}, recibido {window_size}"
        )

    return window_size


def guard_no_future_timestamps(
    timestamps: List[float],
    *,
    tolerance_seconds: float = 300.0,
) -> None:
    """Rechaza timestamps que estén significativamente en el futuro.

    Args:
        timestamps: Lista de timestamps Unix.
        tolerance_seconds: Margen permitido en el futuro (default 5 min).

    Raises:
        InputGuardError: Si algún timestamp excede now + tolerance.
    """
    if not timestamps:
        return

    now = time.time()
    max_allowed = now + tolerance_seconds

    for i, ts in enumerate(timestamps):
        if ts > max_allowed:
            raise InputGuardError(
                f"Timestamp en posición {i} está en el futuro: "
                f"{ts:.0f} > {max_allowed:.0f} (now + {tolerance_seconds}s)"
            )


def safe_series_id_to_int(series_id: str, *, fallback: int = 0) -> int:
    """Safely convert a series_id string to a legacy sensor_id int.

    Used by bridge methods that delegate from agnostic ``series_id: str``
    to legacy ``sensor_id: int`` APIs.  Unlike bare ``int(series_id)``,
    this never raises and logs a warning for non-numeric IDs.

    Args:
        series_id: Agnostic series identifier.
        fallback: Value to return when ``series_id`` is not numeric.

    Returns:
        Integer sensor_id, or ``fallback`` if conversion fails.
    """
    if series_id.isdigit():
        return int(series_id)
    # Try negative / float-like strings
    try:
        return int(series_id)
    except (ValueError, TypeError):
        import logging
        logging.getLogger(__name__).debug(
            "series_id_to_int_fallback",
            extra={"series_id": series_id, "fallback": fallback},
        )
        return fallback


def guard_finite_value(value: object, name: str = "value") -> float:
    """Valida que un valor sea un float finito.

    Args:
        value: Valor a validar.
        name: Nombre del campo para mensajes de error.

    Returns:
        Valor como float finito.

    Raises:
        InputGuardError: Si no es finito.
    """
    if value is None:
        raise InputGuardError(f"{name} es requerido")

    try:
        f = float(value)
    except (TypeError, ValueError):
        raise InputGuardError(
            f"{name} debe ser numérico, recibido {type(value).__name__}"
        )

    if not math.isfinite(f):
        raise InputGuardError(f"{name} debe ser finito, recibido {f}")

    return f
