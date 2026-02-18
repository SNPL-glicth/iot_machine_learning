"""Módulo de Correlación de Sensores - Facade principal.

FALENCIA 2: Cada sensor se procesa de forma independiente, sin correlación.

Este módulo implementa:
- Detección de patrones correlacionados entre sensores del mismo dispositivo
- Identificación de patrones multi-sensor (ej: temp↑ + humedad↓ = falla HVAC)
- Agregación de eventos relacionados
- Detección de anomalías que solo son visibles en conjunto

Modular implementation:
    - types: Entities and enums (CorrelationPattern, SensorSnapshot, etc.)
    - queries: Database queries (get_device_sensors, get_correlated_events)
    - pattern_matcher: Correlation logic (SensorCorrelator class)
    - sensor_correlator: Main facade (this file - backward compatibility)
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy.engine import Connection

from .pattern_matcher import SensorCorrelator
from .types import CorrelationResult


def correlate_sensor_with_device(
    conn: Connection,
    sensor_id: int,
) -> Optional[CorrelationResult]:
    """Función de conveniencia para correlacionar un sensor con su dispositivo.
    
    Esta función es el punto de entrada principal para el análisis de correlación
    desde el batch runner.
    
    Args:
        conn: SQLAlchemy connection
        sensor_id: ID del sensor a correlacionar
        
    Returns:
        CorrelationResult si se detecta un patrón significativo, None si no.
    """
    correlator = SensorCorrelator(conn)
    return correlator.analyze_all_devices_for_sensor(sensor_id)
