"""Threshold validator service for ML online processing.

Extraído de ml_stream_runner.py para modularidad.
Responsabilidad: Validar umbrales y estado operacional del sensor.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Dict, Optional

from sqlalchemy import text

from iot_ingest_services.common.db import get_engine
from iot_ingest_services.ingest_api.sensor_state import SensorStateManager

logger = logging.getLogger(__name__)


class ThresholdValidator:
    """Valida umbrales y estado operacional del sensor.
    
    FIX P0-2: Cache en memoria para thresholds WARNING con TTL configurable.
    
    Responsabilidades:
    - Verificar si el sensor puede emitir eventos (NO cacheado — estado dinámico)
    - Verificar si el valor está dentro del rango WARNING del usuario (cacheado)
    - Cargar umbrales desde la BD con stale-while-revalidate
    
    REGLA DE DOMINIO CRÍTICA:
    Si el valor está dentro del rango WARNING del usuario, el ML NO puede
    generar eventos WARNING o CRITICAL. El usuario definió ese rango como
    "normal" y el ML debe respetarlo.
    """
    
    def __init__(self) -> None:
        self._thresholds_cache: Dict[int, dict] = {}
        self._cache_timestamps: Dict[int, float] = {}
        self._cache_ttl_seconds = float(
            os.getenv("ML_THRESHOLD_CACHE_TTL_SECONDS", "30.0")
        )
        self._cache_lock = threading.Lock()
    
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

    def invalidate_cache(self, sensor_id: int) -> None:
        """Invalida el cache de thresholds para un sensor específico."""
        with self._cache_lock:
            self._thresholds_cache.pop(sensor_id, None)
            self._cache_timestamps.pop(sensor_id, None)
        logger.debug(
            "ml_threshold_cache_invalidate",
            extra={"sensor_id": sensor_id},
        )

    def invalidate_all(self) -> None:
        """Invalida todo el cache de thresholds."""
        with self._cache_lock:
            self._thresholds_cache.clear()
            self._cache_timestamps.clear()
        logger.debug("ml_threshold_cache_invalidate_all")

    def _load_warning_thresholds(self, sensor_id: int) -> dict:
        """Carga los umbrales WARNING del sensor desde la BD con cache.

        FIX P0-2:
        - Cache en memoria con TTL configurable (ML_THRESHOLD_CACHE_TTL_SECONDS).
        - Stale-while-revalidate: si la query falla, usa el valor cacheado anterior.
        - Thread-safe para soportar paralelización futura.

        Returns:
            dict con warning_min y warning_max, o dict vacío si no hay umbrales
        """
        now = time.monotonic()

        with self._cache_lock:
            cached = self._thresholds_cache.get(sensor_id)
            cached_ts = self._cache_timestamps.get(sensor_id)
            if cached is not None and cached_ts is not None:
                age = now - cached_ts
                if age < self._cache_ttl_seconds:
                    logger.debug(
                        "ml_threshold_cache_hit",
                        extra={
                            "sensor_id": sensor_id,
                            "age_seconds": round(age, 2),
                            "ttl_seconds": self._cache_ttl_seconds,
                        },
                    )
                    return cached
                # TTL expirado — intentar refrescar, pero conservar stale para fallback
                stale_value = dict(cached) if cached else None
            else:
                stale_value = None

        # Cache miss o expirado: consultar BD (fuera del lock para no bloquear)
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

                result = {}
                if row:
                    if row[0] is not None:
                        result["warning_min"] = float(row[0])
                    if row[1] is not None:
                        result["warning_max"] = float(row[1])

                with self._cache_lock:
                    self._thresholds_cache[sensor_id] = result
                    self._cache_timestamps[sensor_id] = now

                logger.debug(
                    "ml_threshold_cache_miss",
                    extra={
                        "sensor_id": sensor_id,
                        "has_thresholds": bool(result),
                    },
                )
                return result

        except Exception as e:
            logger.warning(
                "ml_threshold_load_error",
                extra={
                    "sensor_id": sensor_id,
                    "error": str(e),
                    "stale_available": stale_value is not None,
                },
            )
            # Stale-while-revalidate: usar cache anterior si existe
            if stale_value is not None:
                logger.info(
                    "ml_threshold_cache_stale_fallback",
                    extra={
                        "sensor_id": sensor_id,
                        "stale_age_seconds": round(now - (self._cache_timestamps.get(sensor_id, now)), 2),
                    },
                )
                return stale_value

            return {}
