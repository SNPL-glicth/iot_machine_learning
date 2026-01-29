"""Threshold validator service for ML online processing.

Extraído de ml_stream_runner.py para modularidad.
Responsabilidad: Validar umbrales y estado operacional del sensor.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from sqlalchemy import text

from iot_ingest_services.common.db import get_engine
from iot_ingest_services.ingest_api.sensor_state import SensorStateManager

logger = logging.getLogger(__name__)


class ThresholdValidator:
    """Valida umbrales y estado operacional del sensor.
    
    Responsabilidades:
    - Verificar si el sensor puede emitir eventos
    - Verificar si el valor está dentro del rango WARNING del usuario
    - Cargar umbrales desde la BD
    
    REGLA DE DOMINIO CRÍTICA:
    Si el valor está dentro del rango WARNING del usuario, el ML NO puede
    generar eventos WARNING o CRITICAL. El usuario definió ese rango como
    "normal" y el ML debe respetarlo.
    """
    
    def __init__(self) -> None:
        self._thresholds_cache: Dict[int, dict] = {}
    
    def can_sensor_emit_events(self, sensor_id: int) -> bool:
        """Verifica si el sensor puede emitir eventos ML.
        
        REGLA DE DOMINIO:
        - Sensores en INITIALIZING o STALE NO pueden generar eventos
        - Solo sensores en NORMAL, WARNING o ALERT pueden generar eventos
        
        Returns:
            True si el sensor puede emitir eventos, False en caso contrario
        """
        try:
            engine = get_engine()
            with engine.connect() as conn:
                state_manager = SensorStateManager(conn)
                can_generate, reason = state_manager.can_generate_events(sensor_id)
                if not can_generate:
                    logger.debug(
                        "[ML_BLOCKED] sensor_id=%s cannot emit events: %s",
                        sensor_id, reason
                    )
                return can_generate
        except Exception as e:
            logger.warning(
                "[ML_STATE_CHECK_ERROR] sensor_id=%s error=%s, allowing events",
                sensor_id, str(e)
            )
            # En caso de error, permitir eventos (fail-open para no bloquear ML)
            return True

    def is_value_within_warning_range(self, sensor_id: int, value: float) -> bool:
        """Verifica si el valor está dentro del rango WARNING definido por el usuario.
        
        REGLA DE DOMINIO CRÍTICA:
        Si el valor está dentro de [warning_min, warning_max], el ML NO puede
        generar eventos WARNING o CRITICAL. El usuario definió ese rango como
        "normal" y el ML debe respetarlo.
        
        Args:
            sensor_id: ID del sensor
            value: Valor actual de la lectura
            
        Returns:
            True si el valor está dentro del rango WARNING (ML no debe alertar)
            False si el valor está fuera del rango o no hay umbrales configurados
        """
        thresholds = self._load_warning_thresholds(sensor_id)
        
        if not thresholds:
            logger.debug(
                "[ML_THRESHOLD_CHECK] sensor_id=%s value=%.4f NO_THRESHOLDS - ML puede operar",
                sensor_id, value
            )
            return False
        
        warning_min = thresholds.get("warning_min")
        warning_max = thresholds.get("warning_max")
        
        if warning_min is None and warning_max is None:
            logger.debug(
                "[ML_THRESHOLD_CHECK] sensor_id=%s value=%.4f NO_LIMITS - ML puede operar",
                sensor_id, value
            )
            return False
        
        within_range = True
        if warning_min is not None and value < warning_min:
            within_range = False
        if warning_max is not None and value > warning_max:
            within_range = False
        
        logger.info(
            "[ML_THRESHOLD_CHECK] sensor_id=%s value=%.4f min=%s max=%s within_range=%s",
            sensor_id, value, warning_min, warning_max, within_range
        )
        
        return within_range

    def _load_warning_thresholds(self, sensor_id: int) -> dict:
        """Carga los umbrales WARNING del sensor desde la BD.
        
        Returns:
            dict con warning_min y warning_max, o dict vacío si no hay umbrales
        """
        try:
            engine = get_engine()
            with engine.connect() as conn:
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
                    return {}
                
                result = {}
                if row[0] is not None:
                    result["warning_min"] = float(row[0])
                if row[1] is not None:
                    result["warning_max"] = float(row[1])
                
                return result
        except Exception as e:
            logger.warning(
                "[ML_THRESHOLD_LOAD_ERROR] sensor_id=%s error=%s",
                sensor_id, str(e)
            )
            return {}
