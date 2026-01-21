"""Módulo de Correlación de Sensores.

FALENCIA 2: Cada sensor se procesa de forma independiente, sin correlación.

Este módulo implementa:
- Detección de patrones correlacionados entre sensores del mismo dispositivo
- Identificación de patrones multi-sensor (ej: temp↑ + humedad↓ = falla HVAC)
- Agregación de eventos relacionados
- Detección de anomalías que solo son visibles en conjunto
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

logger = logging.getLogger(__name__)


class CorrelationPattern(Enum):
    """Patrones de correlación conocidos entre sensores."""
    HVAC_FAILURE = "hvac_failure"           # temp↑ + humedad↓
    HVAC_OVERLOAD = "hvac_overload"         # temp↑ + power↑
    WATER_LEAK = "water_leak"               # humedad↑ + temp↓
    ELECTRICAL_ISSUE = "electrical_issue"   # voltage↓ + power fluctuation
    VENTILATION_BLOCKED = "ventilation_blocked"  # temp↑ + air_quality↓
    UNKNOWN = "unknown"


@dataclass
class SensorSnapshot:
    """Snapshot del estado actual de un sensor."""
    sensor_id: int
    sensor_type: str
    current_value: float
    predicted_value: float
    trend: str  # rising, falling, stable
    anomaly_score: float
    severity: str
    timestamp: datetime


@dataclass
class CorrelationResult:
    """Resultado de análisis de correlación entre sensores."""
    device_id: int
    device_name: str
    pattern_detected: Optional[CorrelationPattern]
    pattern_confidence: float
    correlated_sensors: list[SensorSnapshot]
    combined_severity: str
    description: str
    root_cause_hypothesis: Optional[str]
    is_significant: bool  # True si la correlación es significativa
    
    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "pattern_detected": self.pattern_detected.value if self.pattern_detected else None,
            "pattern_confidence": self.pattern_confidence,
            "correlated_sensors": [
                {
                    "sensor_id": s.sensor_id,
                    "sensor_type": s.sensor_type,
                    "current_value": s.current_value,
                    "predicted_value": s.predicted_value,
                    "trend": s.trend,
                    "anomaly_score": s.anomaly_score,
                    "severity": s.severity,
                }
                for s in self.correlated_sensors
            ],
            "combined_severity": self.combined_severity,
            "description": self.description,
            "root_cause_hypothesis": self.root_cause_hypothesis,
            "is_significant": self.is_significant,
        }


@dataclass
class DeviceSensorGroup:
    """Grupo de sensores de un dispositivo."""
    device_id: int
    device_name: str
    sensors: list[SensorSnapshot] = field(default_factory=list)


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
        
        rows = self._conn.execute(
            text("""
                SELECT 
                    s.id AS sensor_id,
                    s.sensor_type,
                    d.id AS device_id,
                    d.name AS device_name,
                    p.predicted_value,
                    p.trend,
                    p.anomaly_score,
                    p.severity,
                    p.predicted_at,
                    (SELECT TOP 1 sr.value 
                     FROM dbo.sensor_readings sr 
                     WHERE sr.sensor_id = s.id 
                     ORDER BY sr.timestamp DESC) AS current_value
                FROM dbo.sensors s
                JOIN dbo.devices d ON d.id = s.device_id
                LEFT JOIN dbo.predictions p ON p.sensor_id = s.id
                    AND p.id = (
                        SELECT TOP 1 p2.id 
                        FROM dbo.predictions p2 
                        WHERE p2.sensor_id = s.id 
                        ORDER BY p2.predicted_at DESC
                    )
                WHERE s.device_id = :device_id
                  AND s.is_active = 1
            """),
            {"device_id": device_id},
        ).fetchall()
        
        device_name = ""
        sensors = []
        
        for row in rows:
            device_name = row.device_name or f"Device {device_id}"
            
            if row.predicted_value is not None:
                sensors.append(SensorSnapshot(
                    sensor_id=int(row.sensor_id),
                    sensor_type=str(row.sensor_type or "unknown").lower(),
                    current_value=float(row.current_value) if row.current_value else 0.0,
                    predicted_value=float(row.predicted_value),
                    trend=str(row.trend or "stable").lower(),
                    anomaly_score=float(row.anomaly_score) if row.anomaly_score else 0.0,
                    severity=str(row.severity or "info").lower(),
                    timestamp=row.predicted_at or datetime.now(timezone.utc),
                ))
        
        group = DeviceSensorGroup(
            device_id=device_id,
            device_name=device_name,
            sensors=sensors,
        )
        
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
        severities = [s.severity for s in group.sensors]
        if "critical" in severities:
            combined_severity = "critical"
        elif "warning" in severities:
            combined_severity = "warning"
        else:
            combined_severity = "info"
        
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
        
        # Obtener device_id del sensor
        row = self._conn.execute(
            text("SELECT device_id FROM dbo.sensors WHERE id = :sensor_id"),
            {"sensor_id": sensor_id},
        ).fetchone()
        
        if not row:
            return None
        
        device_id = int(row[0])
        return self.analyze_device_correlation(device_id)
    
    def get_correlated_events(
        self,
        sensor_id: int,
        time_window_minutes: int = 30,
    ) -> list[dict]:
        """Obtiene eventos ML recientes de sensores correlacionados (mismo dispositivo)."""
        
        rows = self._conn.execute(
            text("""
                SELECT 
                    e.id AS event_id,
                    e.sensor_id,
                    s.sensor_type,
                    e.event_type,
                    e.event_code,
                    e.title,
                    e.created_at
                FROM dbo.ml_events e
                JOIN dbo.sensors s ON s.id = e.sensor_id
                WHERE s.device_id = (
                    SELECT device_id FROM dbo.sensors WHERE id = :sensor_id
                )
                AND e.sensor_id != :sensor_id
                AND e.created_at >= DATEADD(minute, -:minutes, GETDATE())
                AND e.status IN ('active', 'acknowledged')
                ORDER BY e.created_at DESC
            """),
            {"sensor_id": sensor_id, "minutes": time_window_minutes},
        ).fetchall()
        
        return [
            {
                "event_id": int(r.event_id),
                "sensor_id": int(r.sensor_id),
                "sensor_type": str(r.sensor_type or "unknown"),
                "event_type": str(r.event_type or "notice"),
                "event_code": str(r.event_code or ""),
                "title": str(r.title or ""),
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
    
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


def correlate_sensor_with_device(
    conn: Connection,
    sensor_id: int,
) -> Optional[CorrelationResult]:
    """Función de conveniencia para correlacionar un sensor con su dispositivo.
    
    Esta función es el punto de entrada principal para el análisis de correlación
    desde el batch runner.
    """
    correlator = SensorCorrelator(conn)
    return correlator.analyze_all_devices_for_sensor(sensor_id)
