"""Lógica de detección de patrones de correlación entre sensores."""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.engine import Connection

from .queries import get_correlated_events, get_device_id_for_sensor, get_device_sensors
from .types import (
    CorrelationPattern,
    CorrelationResult,
    DeviceSensorGroup,
    SensorSnapshot,
)

logger = logging.getLogger(__name__)


class SensorCorrelator:
    """Correlacionador de sensores para detectar patrones multi-sensor.
    
    REGLAS DE CORRELACIÓN:
    1. Solo correlaciona sensores del mismo dispositivo
    2. Busca patrones conocidos (HVAC, eléctrico, etc.)
    3. Calcula severidad combinada (peor caso)
    4. Genera hipótesis de causa raíz
    """
    
    # Patrones de correlación conocidos
    # Formato: {(sensor_type1, trend1, sensor_type2, trend2): (pattern, confidence, hypothesis)}
    KNOWN_PATTERNS = {
        ("temperature", "rising", "humidity", "falling"): (
            CorrelationPattern.HVAC_FAILURE,
            0.85,
            "Posible falla en sistema de aire acondicionado (compresor o refrigerante)",
        ),
        ("temperature", "rising", "power", "rising"): (
            CorrelationPattern.HVAC_OVERLOAD,
            0.75,
            "Sistema de climatización trabajando al límite, posible sobrecarga",
        ),
        ("humidity", "rising", "temperature", "falling"): (
            CorrelationPattern.WATER_LEAK,
            0.70,
            "Posible filtración de agua o condensación excesiva",
        ),
        ("voltage", "falling", "power", "rising"): (
            CorrelationPattern.ELECTRICAL_ISSUE,
            0.80,
            "Problema eléctrico: caída de tensión con aumento de consumo",
        ),
        ("temperature", "rising", "air_quality", "falling"): (
            CorrelationPattern.VENTILATION_BLOCKED,
            0.75,
            "Ventilación deficiente: acumulación de calor y CO2",
        ),
    }
    
    def __init__(self, conn: Connection):
        self._conn = conn
        self._sensor_cache: dict[int, DeviceSensorGroup] = {}
    
    def get_device_sensors(self, device_id: int) -> DeviceSensorGroup:
        """Obtiene todos los sensores activos de un dispositivo con sus últimas predicciones."""
        if device_id in self._sensor_cache:
            return self._sensor_cache[device_id]
        
        group = get_device_sensors(self._conn, device_id)
        self._sensor_cache[device_id] = group
        return group
    
    def analyze_device_correlation(self, device_id: int) -> Optional[CorrelationResult]:
        """Analiza correlaciones entre sensores de un dispositivo.
        
        Returns:
            CorrelationResult si se detecta un patrón significativo, None si no.
        """
        group = self.get_device_sensors(device_id)
        
        if len(group.sensors) < 2:
            return None
        
        # Buscar patrones conocidos
        best_pattern = None
        best_confidence = 0.0
        best_hypothesis = None
        correlated_sensors = []
        
        for i, s1 in enumerate(group.sensors):
            for s2 in group.sensors[i + 1:]:
                # Crear clave de patrón (ordenada alfabéticamente para consistencia)
                if s1.sensor_type < s2.sensor_type:
                    key = (s1.sensor_type, s1.trend, s2.sensor_type, s2.trend)
                    pair = [s1, s2]
                else:
                    key = (s2.sensor_type, s2.trend, s1.sensor_type, s1.trend)
                    pair = [s2, s1]
                
                if key in self.KNOWN_PATTERNS:
                    pattern, confidence, hypothesis = self.KNOWN_PATTERNS[key]
                    
                    # Ajustar confianza según anomaly_score de los sensores
                    avg_anomaly = (s1.anomaly_score + s2.anomaly_score) / 2
                    adjusted_confidence = confidence * (1 + avg_anomaly * 0.2)
                    adjusted_confidence = min(0.99, adjusted_confidence)
                    
                    if adjusted_confidence > best_confidence:
                        best_pattern = pattern
                        best_confidence = adjusted_confidence
                        best_hypothesis = hypothesis
                        correlated_sensors = pair
        
        # Calcular severidad combinada
        combined_severity = self._compute_combined_severity(group.sensors)
        
        # Determinar si la correlación es significativa
        is_significant = (
            best_pattern is not None 
            and best_confidence >= 0.6
            and combined_severity in ("warning", "critical")
        )
        
        # Generar descripción
        if best_pattern:
            description = self._generate_correlation_description(
                pattern=best_pattern,
                sensors=correlated_sensors,
                device_name=group.device_name,
            )
        else:
            description = f"No se detectaron patrones de correlación significativos en {group.device_name}"
        
        return CorrelationResult(
            device_id=device_id,
            device_name=group.device_name,
            pattern_detected=best_pattern,
            pattern_confidence=best_confidence,
            correlated_sensors=correlated_sensors if correlated_sensors else group.sensors,
            combined_severity=combined_severity,
            description=description,
            root_cause_hypothesis=best_hypothesis,
            is_significant=is_significant,
        )
    
    def analyze_all_devices_for_sensor(self, sensor_id: int) -> Optional[CorrelationResult]:
        """Analiza correlaciones para el dispositivo que contiene el sensor dado."""
        device_id = get_device_id_for_sensor(self._conn, sensor_id)
        
        if not device_id:
            return None
        
        return self.analyze_device_correlation(device_id)
    
    def get_correlated_events(
        self,
        sensor_id: int,
        time_window_minutes: int = 30,
    ) -> list[dict]:
        """Obtiene eventos ML recientes de sensores correlacionados (mismo dispositivo)."""
        return get_correlated_events(self._conn, sensor_id, time_window_minutes)
    
    def _compute_combined_severity(self, sensors: list[SensorSnapshot]) -> str:
        """Calcula la severidad combinada (peor caso)."""
        severities = [s.severity for s in sensors]
        if "critical" in severities:
            return "critical"
        elif "warning" in severities:
            return "warning"
        else:
            return "info"
    
    def _generate_correlation_description(
        self,
        pattern: CorrelationPattern,
        sensors: list[SensorSnapshot],
        device_name: str,
    ) -> str:
        """Genera una descripción legible del patrón de correlación."""
        sensor_desc = " y ".join([
            f"{s.sensor_type} ({s.trend})"
            for s in sensors
        ])
        
        pattern_names = {
            CorrelationPattern.HVAC_FAILURE: "Posible falla de HVAC",
            CorrelationPattern.HVAC_OVERLOAD: "Sobrecarga de climatización",
            CorrelationPattern.WATER_LEAK: "Posible filtración de agua",
            CorrelationPattern.ELECTRICAL_ISSUE: "Problema eléctrico",
            CorrelationPattern.VENTILATION_BLOCKED: "Ventilación deficiente",
            CorrelationPattern.UNKNOWN: "Patrón desconocido",
        }
        
        pattern_name = pattern_names.get(pattern, "Patrón detectado")
        
        return (
            f"{pattern_name} detectado en {device_name}. "
            f"Sensores correlacionados: {sensor_desc}."
        )
    
    def clear_cache(self) -> None:
        """Limpia el cache de sensores."""
        self._sensor_cache.clear()
