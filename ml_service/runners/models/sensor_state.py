"""Sensor state model for ML online processing."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SensorState:
    """Estado del sensor según análisis ML.
    
    Representa el estado actual de un sensor desde la perspectiva del ML:
    - severity: Nivel de severidad (NORMAL, WARN, CRITICAL)
    - recommended_action: Acción recomendada
    - behavior_pattern: Patrón de comportamiento detectado
    - in_transient_anomaly: Si está en anomalía transitoria
    - last_event_code: Último código de evento emitido
    """
    
    severity: str
    recommended_action: str
    behavior_pattern: str
    in_transient_anomaly: bool
    last_event_code: Optional[str] = None
