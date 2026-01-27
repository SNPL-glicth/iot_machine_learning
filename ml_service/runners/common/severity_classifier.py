"""Clasificador de severidad.

Responsabilidad única: Clasificar severidad combinando anomalía + riesgo físico.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import text
from sqlalchemy.engine import Connection

if TYPE_CHECKING:
    from iot_machine_learning.ml_service.repository.sensor_repository import SensorMetadata
    from iot_machine_learning.ml_service.models.regression_model import Trend

logger = logging.getLogger(__name__)


@dataclass
class SeverityResult:
    """Resultado de clasificación de severidad."""
    risk_level: str
    severity: str
    action_required: bool
    recommended_action: str


class SeverityClassifier:
    """Clasifica severidad combinando anomalía estadística y riesgo físico.
    
    Prioridad de reglas:
    1) Violación de umbral físico => CRITICAL
    2) Anomalía + riesgo alto => CRITICAL
    3) Anomalía o riesgo alto => WARNING
    4) Resto => INFO
    """
    
    # Rangos recomendados por tipo de sensor (fallback)
    DEFAULT_RANGES = {
        "temperature": (15.0, 35.0),
        "humidity": (30.0, 70.0),
        "air_quality": (400.0, 1000.0),
        "power": (0.0, 100000.0),
        "voltage": (0.0, 100000.0),
    }
    
    def get_user_defined_range(
        self, 
        conn: Connection, 
        sensor_id: int
    ) -> tuple[float, float] | None:
        """Obtiene el rango definido por el usuario desde alert_thresholds.
        
        REGLA DE DOMINIO: Los umbrales del usuario tienen PRIORIDAD.
        """
        row = conn.execute(
            text(
                """
                SELECT 
                    threshold_value_min,
                    threshold_value_max
                FROM dbo.alert_thresholds
                WHERE sensor_id = :sensor_id
                  AND is_active = 1
                  AND condition_type = 'out_of_range'
                ORDER BY 
                    CASE severity WHEN 'warning' THEN 0 ELSE 1 END,
                    id ASC
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()
        
        if not row:
            return None
        
        min_val = float(row[0]) if row[0] is not None else None
        max_val = float(row[1]) if row[1] is not None else None
        
        if min_val is None and max_val is None:
            return None
        
        # Si solo hay un límite, usar rango muy amplio para el otro
        if min_val is None:
            min_val = float('-inf')
        if max_val is None:
            max_val = float('inf')
        
        return (min_val, max_val)
    
    def is_value_within_user_thresholds(
        self,
        conn: Connection,
        sensor_id: int,
        value: float,
    ) -> bool:
        """Verifica si el valor está dentro de los umbrales WARNING del usuario.
        
        REGLA DE DOMINIO: Si está dentro del rango, ML NO debe alertar.
        """
        row = conn.execute(
            text(
                """
                SELECT 
                    threshold_value_min,
                    threshold_value_max
                FROM dbo.alert_thresholds
                WHERE sensor_id = :sensor_id
                  AND is_active = 1
                  AND severity = 'warning'
                  AND condition_type = 'out_of_range'
                ORDER BY id ASC
                """
            ),
            {"sensor_id": sensor_id},
        ).fetchone()
        
        if not row:
            return False
        
        warning_min = float(row[0]) if row[0] is not None else None
        warning_max = float(row[1]) if row[1] is not None else None
        
        if warning_min is None and warning_max is None:
            return False
        
        if warning_min is not None and value < warning_min:
            return False
        if warning_max is not None and value > warning_max:
            return False
        
        return True
    
    def derive_recommended_range(
        self, 
        sensor_type: str
    ) -> tuple[float, float] | None:
        """Heurística de rango recomendado por tipo de sensor (fallback)."""
        return self.DEFAULT_RANGES.get(sensor_type)
    
    def compute_risk_level(
        self, 
        sensor_type: str, 
        predicted_value: float
    ) -> str:
        """Clasifica nivel de riesgo físico.
        
        Returns: 'LOW' | 'MEDIUM' | 'HIGH' | 'NONE'
        """
        rng = self.derive_recommended_range(sensor_type)
        if rng is None:
            return "NONE"

        min_ok, max_ok = rng
        if min_ok <= predicted_value <= max_ok:
            return "LOW"

        margin = 0.1 * (max_ok - min_ok)
        if predicted_value < min_ok - margin or predicted_value > max_ok + margin:
            return "HIGH"

        return "MEDIUM"
    
    def compute_severity(
        self, 
        *, 
        is_anomaly: bool, 
        risk_level: str, 
        out_of_physical_range: bool
    ) -> str:
        """Combina anomalía + riesgo en severidad única."""
        rl = (risk_level or "").upper()

        if out_of_physical_range:
            return "critical"
        if is_anomaly and rl == "HIGH":
            return "critical"
        if is_anomaly or rl == "HIGH":
            return "warning"
        return "info"
    
    def classify(
        self,
        *,
        sensor_type: str,
        location: str,
        predicted_value: float,
        trend: str,
        anomaly: bool,
        anomaly_score: float,
        confidence: float,
        horizon_minutes: int,
        user_defined_range: tuple[float, float] | None = None,
    ) -> SeverityResult:
        """Clasifica severidad completa.
        
        Returns:
            SeverityResult con risk_level, severity, action_required, recommended_action
        """
        risk_level = self.compute_risk_level(sensor_type, predicted_value)

        # Usar umbrales del usuario si están disponibles
        rng = user_defined_range if user_defined_range else self.derive_recommended_range(sensor_type)
        out_of_range = False
        if rng is not None:
            min_ok, max_ok = rng
            out_of_range = predicted_value < min_ok or predicted_value > max_ok

        severity = self.compute_severity(
            is_anomaly=anomaly,
            risk_level=risk_level,
            out_of_physical_range=out_of_range,
        )

        action_required = False
        recommended_action = "Sin acción requerida. Seguir monitoreando."

        if severity == "info":
            rl = risk_level.upper()
            if rl in {"MEDIUM", "HIGH"}:
                recommended_action = (
                    f"La proyección se acerca a los límites operativos en {location}. "
                    "Supervisar en las próximas horas."
                )
            else:
                recommended_action = "Valores dentro del rango esperado. No se requiere acción."
            return SeverityResult(risk_level, severity, False, recommended_action)

        action_required = True
        rl = risk_level.upper()

        if severity == "critical":
            recommended_action = (
                f"Condición crítica en {location}. "
                "Revisar inmediatamente el equipo y condiciones ambientales."
            )
        else:  # warning
            if rl == "HIGH":
                recommended_action = (
                    f"Riesgo elevado en {location}. "
                    "Programar revisión prioritaria."
                )
            else:
                recommended_action = (
                    f"Comportamiento inusual en {location}. "
                    "Supervisar de cerca."
                )

        return SeverityResult(risk_level, severity, action_required, recommended_action)
