"""Contexto semántico inyectable — Nivel 2 (Semántico).

UTSAE primero analiza la estructura matemática (Nivel 1).
Después se inyecta contexto para interpretar los hallazgos.

SeriesContext NO participa en cálculos matemáticos.
Solo se usa para:
1. Enriquecer narrativas ("en el contexto de energía esto implica...")
2. Aplicar reglas de negocio (umbrales, severidad)
3. Decidir acciones (alertar, ignorar, escalar)

Diseño agnóstico:
- domain_name: "iot", "finance", "network", "health", etc.
- entity_type: "temperature_sensor", "stock_price", "latency_probe", etc.
- thresholds: dict genérico, no hardcoded a IoT.
- business_rules: dict genérico para reglas del dominio.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class Threshold:
    """Umbral genérico con niveles de severidad.

    Attributes:
        warning_low: Límite inferior de warning (None = sin límite).
        warning_high: Límite superior de warning.
        critical_low: Límite inferior de critical.
        critical_high: Límite superior de critical.
    """

    warning_low: Optional[float] = None
    warning_high: Optional[float] = None
    critical_low: Optional[float] = None
    critical_high: Optional[float] = None

    def is_within_normal(self, value: float) -> bool:
        """True si el valor está dentro del rango normal (no warning ni critical)."""
        if self.warning_low is not None and value < self.warning_low:
            return False
        if self.warning_high is not None and value > self.warning_high:
            return False
        return True

    def severity_for(self, value: float) -> str:
        """Retorna 'normal', 'warning' o 'critical' según el valor."""
        if self.critical_low is not None and value < self.critical_low:
            return "critical"
        if self.critical_high is not None and value > self.critical_high:
            return "critical"
        if self.warning_low is not None and value < self.warning_low:
            return "warning"
        if self.warning_high is not None and value > self.warning_high:
            return "warning"
        return "normal"


@dataclass(frozen=True)
class SeriesContext:
    """Contexto semántico de una serie temporal — se inyecta en Nivel 2.

    Attributes:
        domain_name: Dominio de aplicación ("iot", "finance", "network", etc.).
        entity_type: Tipo de entidad que genera la serie ("temperature_sensor",
            "stock_price", "cpu_latency", etc.).
        entity_id: Identificador de la entidad en el dominio (sensor_id, ticker, etc.).
        unit: Unidad de medida ("°C", "USD", "ms", etc.).
        description: Descripción legible de la serie.
        threshold: Umbrales configurados por el usuario/negocio.
        business_rules: Reglas de negocio específicas del dominio.
        metadata: Metadata adicional libre.
    """

    domain_name: str = "unknown"
    entity_type: str = "unknown"
    entity_id: str = ""
    unit: str = ""
    description: str = ""
    threshold: Optional[Threshold] = None
    business_rules: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_iot_sensor(
        cls,
        sensor_id: int,
        sensor_type: str = "",
        device_id: Optional[int] = None,
        unit: str = "",
        threshold: Optional[Threshold] = None,
        **extra: Any,
    ) -> SeriesContext:
        """Factory method para contexto IoT (backward compatible).

        Esto es un bridge de conveniencia para el dominio IoT existente.
        UTSAE no depende de este método — es solo un helper.
        """
        return cls(
            domain_name="iot",
            entity_type=sensor_type or "sensor",
            entity_id=str(sensor_id),
            unit=unit,
            description=f"Sensor {sensor_id} ({sensor_type})" if sensor_type else f"Sensor {sensor_id}",
            threshold=threshold,
            metadata={"device_id": device_id, **extra},
        )
