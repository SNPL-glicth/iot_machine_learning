"""OperationalRegime — régimen operacional de una serie temporal.

Entidad pura del dominio — sin dependencias de infraestructura.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OperationalRegime:
    """Régimen operacional de una serie.

    Representa un "modo" de operación (idle, activo, pico, enfriamiento).

    Attributes:
        regime_id: Identificador numérico del régimen.
        name: Nombre legible (``"idle"``, ``"active"``, ``"peak"``).
        mean_value: Valor medio típico en este régimen.
        std_value: Desviación estándar típica.
        typical_duration_s: Duración promedio en segundos.
    """

    regime_id: int
    name: str
    mean_value: float
    std_value: float
    typical_duration_s: float = 0.0
