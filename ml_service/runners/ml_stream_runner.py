from __future__ import annotations

"""Runner sencillo para consumo online de lecturas por ML.

Este módulo demuestra cómo el ML online depende solo de la interfaz
`ReadingBroker` y de la base de datos, sin conocer la implementación
concreta del broker ni detalles de transporte.

En un MVP puedes ejecutarlo con el broker en memoria, y en el futuro
cambiar a otro broker sin tocar la lógica de ML.
"""

import logging
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy import text
from sqlalchemy.engine import Connection

# Imports de infraestructura compartida (BD)
from iot_ingest_services.common.db import get_engine

# Imports internos de ML (ahora en iot_machine_learning)
from iot_machine_learning.ml_service.config.ml_config import (
    DEFAULT_ML_CONFIG,
    OnlineBehaviorConfig,
)
from iot_machine_learning.ml_service.reading_broker import Reading, ReadingBroker
from iot_machine_learning.ml_service.sliding_window_buffer import (
    SlidingWindowBuffer,
    WindowStats,
)
from iot_machine_learning.ml_service.repository.sensor_repository import (
    get_device_id_for_sensor,
)

# FIX CRÍTICO: Importar validación de umbrales y estado operacional
# El ML NO puede generar eventos si el valor está dentro del rango WARNING del usuario
from iot_ingest_services.ingest_api.sensor_state import SensorStateManager, SensorOperationalState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Estado por sensor y análisis de comportamiento
# ---------------------------------------------------------------------------


@dataclass
class SensorState:
    severity: str
    recommended_action: str
    behavior_pattern: str
    in_transient_anomaly: bool
    last_event_code: Optional[str] = None


@dataclass
class OnlineAnalysis:
    behavior_pattern: str
    is_curve_anomalous: bool
    has_microvariation: bool
    microvariation_delta: float
    new_transient_anomaly: bool
    recovered_transient: bool
    baseline_mean: float
    baseline_std: float
    last_value: float
    z_score_last: float
    slope_short: float
    slope_medium: float
    slope_long: float
    accel_short_vs_medium: float
    accel_medium_vs_long: float


class SimpleMlOnlineProcessor:
    """Procesador online usando SlidingWindowBuffer.

    - Mantiene ventanas deslizantes 1s/5s/10s por sensor.
    - Calcula agregados (avg, min, max, trend, std_dev) por ventana.
    - Modela el COMPORTAMIENTO del sensor (patrón + anomalías).
    - Solo inserta en ml_events cuando cambia el estado o el tipo de evento.
    
    FIX CRÍTICO: El ML NO genera eventos WARNING/CRITICAL si:
    1. El sensor está en INITIALIZING o STALE
    2. El valor está dentro del rango WARNING definido por el usuario
    
    El ML solo COMPLEMENTA, no decide umbrales ni genera eventos por sí solo.
    """

    def __init__(self, cfg: OnlineBehaviorConfig | None = None) -> None:
        self._cfg: OnlineBehaviorConfig = cfg or DEFAULT_ML_CONFIG.online
        self._last_state: Dict[int, SensorState] = {}
        self._buffer = SlidingWindowBuffer(max_horizon_seconds=10.0)
        self._device_cache: Dict[int, int] = {}
        # FIX CRÍTICO: Cache de umbrales por sensor para validación
        self._thresholds_cache: Dict[int, dict] = {}

    def handle_reading(self, reading: Reading) -> None:
        """Procesa una lectura y emite eventos en BD cuando cambia el estado.

        El procesamiento es puramente online: ventanas deslizantes en memoria y
        escritura de eventos en `ml_events` + `alert_notifications`.
        
        FIX CRÍTICO: Antes de emitir eventos, verifica:
        1. Estado operacional del sensor (INITIALIZING/STALE no generan eventos)
        2. Si el valor está dentro del rango WARNING del usuario (no generar eventos)
        """
        import time
        start_time = time.time()

        sensor_id = reading.sensor_id
        reading_ts = float(reading.timestamp)
        ingest_latency_ms = (start_time - reading_ts) * 1000 if reading_ts > 0 else 0
        
        # FIX Issue 3: Log latency for debugging ML warning delays
        if ingest_latency_ms > 1000:  # Log if latency > 1 second
            logger.warning(
                "[ML_LATENCY] High ingest latency sensor_id=%s latency_ms=%.1f",
                sensor_id, ingest_latency_ms
            )
        
        # =========================================================================
        # FIX CRÍTICO: Verificar estado operacional ANTES de procesar
        # =========================================================================
        can_emit_events = self._can_sensor_emit_events(sensor_id)
        if not can_emit_events:
            # Sensor en INITIALIZING o STALE - solo actualizar buffer, no emitir eventos
            self._buffer.add_reading(
                sensor_id=sensor_id,
                value=float(reading.value),
                timestamp=reading_ts,
                windows=(1.0, 5.0, 10.0),
            )
            return
        
        # =========================================================================
        # FIX CRÍTICO: Verificar si valor está dentro del rango WARNING del usuario
        # =========================================================================
        value_within_warning_range = self._is_value_within_warning_range(
            sensor_id, float(reading.value)
        )
        
        stats_by_window = self._buffer.add_reading(
            sensor_id=sensor_id,
            value=float(reading.value),
            timestamp=reading_ts,
            windows=(1.0, 5.0, 10.0),
        )

        if not stats_by_window:
            # Todavía no hay datos suficientes
            return

        prev_state = self._last_state.get(sensor_id)
        analysis = self._analyze_windows(stats_by_window, prev_state)

        # FIX CRÍTICO: Pasar flag de rango al builder para subordinar ML a umbrales
        severity, action, explanation = self._build_explanation(
            sensor_type=reading.sensor_type,
            analysis=analysis,
            value_within_warning_range=value_within_warning_range,
        )

        # Actualizar estado lógico (incluyendo flag de anomalía transitoria)
        prev_in_transient = prev_state.in_transient_anomaly if prev_state else False
        in_transient = prev_in_transient or analysis.new_transient_anomaly
        if analysis.recovered_transient:
            in_transient = False

        new_state = SensorState(
            severity=severity,
            recommended_action=action,
            behavior_pattern=analysis.behavior_pattern,
            in_transient_anomaly=in_transient,
            last_event_code=prev_state.last_event_code if prev_state else None,
        )

        # Decidir si emitimos un evento de estado/comportamiento
        # FIX CRÍTICO: Pasar flag de rango para bloquear eventos de anomalía
        should_emit, event_code = self._should_emit_state_event(
            prev_state=prev_state,
            new_state=new_state,
            analysis=analysis,
            value_within_warning_range=value_within_warning_range,
        )

        # Validación adicional: si el valor está dentro del rango WARNING,
        # asegurar que no se emitan eventos de severidad WARNING o CRITICAL
        if should_emit and value_within_warning_range and severity in ("WARN", "CRITICAL"):
            logger.debug(
                "[ML_SUPPRESSED] sensor_id=%s value=%.4f within warning range, "
                "suppressing %s event",
                sensor_id, float(reading.value), severity
            )
            should_emit = False
            new_state.severity = "NORMAL"

        if should_emit:
            event_type = self._map_severity_to_event_type(severity)
            title = self._build_title(event_code, severity, analysis.behavior_pattern)

            # FIX Issue 3: Log event emission with timing
            processing_time_ms = (time.time() - start_time) * 1000
            logger.info(
                "[ML_EVENT] Emitting event sensor_id=%s code=%s severity=%s processing_ms=%.1f",
                sensor_id, event_code, severity, processing_time_ms
            )

            self._insert_ml_event(
                sensor_id=sensor_id,
                sensor_type=reading.sensor_type,
                severity_label=severity,
                event_type=event_type,
                event_code=event_code,
                title=title,
                explanation=explanation,
                recommended_action=action,
                analysis=analysis,
                ts_utc=float(reading.timestamp),
                prediction_id=None,
                extra_payload=None,
            )

            new_state.last_event_code = event_code

        # Guardar siempre el último estado para analizar patrones / transitorios
        self._last_state[sensor_id] = new_state

        # Autovalidación de predicción vs valor real (evento PREDICTION_DEVIATION)
        try:
            self._check_prediction_deviation(sensor_id=sensor_id, reading=reading, analysis=analysis)
        except Exception:
            logger.exception(
                "Error evaluando desviación de predicción para sensor_id=%s", sensor_id
            )

    # ------------------------------------------------------------------
    # Análisis de ventanas y clasificación de patrón
    # ------------------------------------------------------------------

    def _analyze_windows(
        self,
        stats_by_window: dict[str, WindowStats],
        prev_state: Optional[SensorState],
    ) -> OnlineAnalysis:
        cfg = self._cfg

        w1 = stats_by_window.get("w1")
        w5 = stats_by_window.get("w5")
        w10 = stats_by_window.get("w10")

        baseline = w10 or w5 or w1 or next(iter(stats_by_window.values()))

        baseline_mean = baseline.mean
        baseline_std = baseline.std_dev
        last_value = (w1 or baseline).last_value

        z_score = 0.0
        if baseline_std > 0:
            z_score = (last_value - baseline_mean) / baseline_std

        slope_short = w1.trend if w1 else 0.0
        slope_medium = w5.trend if w5 else slope_short
        slope_long = w10.trend if w10 else slope_medium

        accel_short_vs_medium = slope_short - slope_medium
        accel_medium_vs_long = slope_medium - slope_long

        is_curve_anomalous = any(
            abs(s) >= cfg.slope_anomaly_threshold
            for s in (slope_short, slope_medium, slope_long)
        ) or any(
            abs(a) >= cfg.accel_anomaly_threshold
            for a in (accel_short_vs_medium, accel_medium_vs_long)
        )

        delta = last_value - baseline_mean
        has_microvariation = (
            abs(delta) >= cfg.microvariation_min_delta
            and abs(z_score) >= cfg.microvariation_z_score
            and not is_curve_anomalous
        )

        prev_in_transient = prev_state.in_transient_anomaly if prev_state else False
        outlier_for_transient = abs(z_score) >= cfg.transient_z_score

        new_transient = not prev_in_transient and outlier_for_transient
        recovered = prev_in_transient and not outlier_for_transient

        behavior_pattern = self._classify_pattern(
            w1=w1,
            w5=w5,
            w10=w10,
            baseline_mean=baseline_mean,
            z_score_last=z_score,
            is_curve_anomalous=is_curve_anomalous,
            has_microvariation=has_microvariation,
        )

        return OnlineAnalysis(
            behavior_pattern=behavior_pattern,
            is_curve_anomalous=is_curve_anomalous,
            has_microvariation=has_microvariation,
            microvariation_delta=delta,
            new_transient_anomaly=new_transient,
            recovered_transient=recovered,
            baseline_mean=baseline_mean,
            baseline_std=baseline_std,
            last_value=last_value,
            z_score_last=z_score,
            slope_short=slope_short,
            slope_medium=slope_medium,
            slope_long=slope_long,
            accel_short_vs_medium=accel_short_vs_medium,
            accel_medium_vs_long=accel_medium_vs_long,
        )

    def _classify_pattern(
        self,
        *,
        w1: Optional[WindowStats],
        w5: Optional[WindowStats],
        w10: Optional[WindowStats],
        baseline_mean: float,
        z_score_last: float,
        is_curve_anomalous: bool,
        has_microvariation: bool,
    ) -> str:
        """Clasificación cualitativa del patrón de comportamiento del sensor."""

        # Usamos principalmente la ventana más larga disponible para estabilidad
        ref = w10 or w5 or w1
        if ref is None:
            return "STABLE"

        var_long = ref.std_dev
        trend_long = ref.trend

        # Heurísticas simples por patrón
        if abs(trend_long) < 0.05 and var_long < 0.01 and abs(z_score_last) < 1.0:
            return "STABLE"

        # Oscilación: alta variabilidad y cambios de signo en la pendiente
        if var_long >= 0.05:
            t1 = w1.trend if w1 else trend_long
            t5 = w5.trend if w5 else trend_long
            if t1 * t5 < 0:
                return "OSCILLATING"

        # Drift: tendencia suave pero sostenida
        if 0.05 <= abs(trend_long) < self._cfg.slope_anomaly_threshold and var_long < 0.1:
            return "DRIFTING"

        # Spikes: curva muy anómala pero localizada
        if is_curve_anomalous and has_microvariation and abs(z_score_last) >= self._cfg.microvariation_z_score:
            return "SPIKING"

        # Exponencial (rise/fall aproximado): pendiente creciente en ventanas más cortas
        s_short = w1.trend if w1 else trend_long
        s_med = w5.trend if w5 else trend_long
        if is_curve_anomalous:
            if s_short > s_med and s_med > trend_long and trend_long > 0:
                return "EXPONENTIAL_RISE"
            if s_short < s_med and s_med < trend_long and trend_long < 0:
                return "EXPONENTIAL_FALL"

        # Fallback genérico
        return "DRIFTING" if abs(trend_long) >= 0.05 else "OSCILLATING"

    # ------------------------------------------------------------------
    # Construcción de explicación y decisión de eventos
    # ------------------------------------------------------------------

    def _build_explanation(
        self,
        *,
        sensor_type: str,
        analysis: OnlineAnalysis,
        value_within_warning_range: bool = False,
    ) -> tuple[str, str, str]:
        """Devuelve (severity_label, recommended_action, explanation).
        
        FIX CRÍTICO: Si el valor está dentro del rango WARNING del usuario,
        el ML NO puede generar severidad WARNING/CRITICAL. El usuario definió
        ese rango como "normal" y el ML debe respetarlo.
        
        REGLA DE DOMINIO:
        - Umbrales del usuario > Patrones estadísticos del ML
        - ML solo COMPLEMENTA, no decide umbrales
        """

        pattern = analysis.behavior_pattern

        # =========================================================================
        # FIX CRÍTICO: Si el valor está dentro del rango WARNING del usuario,
        # FORZAR severidad NORMAL independientemente del patrón estadístico.
        # El usuario definió ese rango como aceptable.
        # =========================================================================
        if value_within_warning_range:
            severity = "NORMAL"
            action = "none"
            human = (
                "Valor dentro del rango definido por el usuario. "
                f"Patrón detectado: {pattern} (informativo, sin acción requerida)."
            )
        elif pattern == "STABLE":
            severity = "NORMAL"
            action = "none"
            human = "Comportamiento estable del sensor, dentro de su patrón histórico."
        elif pattern in {"OSCILLATING", "DRIFTING"}:
            severity = "WARN"
            action = "check_soon"
            human = (
                "El sensor muestra oscilaciones o una deriva sostenida; "
                "revisar calibración y condiciones ambientales a la brevedad."
            )
        else:
            severity = "CRITICAL"
            action = "immediate_intervention"
            human = (
                "Comportamiento fuertemente anómalo (spikes o crecimiento/caída exponencial); "
                "se recomienda intervención inmediata sobre el dispositivo."
            )

        trend_desc: str
        if analysis.slope_long > 0.1:
            trend_desc = "en aumento"
        elif analysis.slope_long < -0.1:
            trend_desc = "en descenso"
        else:
            trend_desc = "estable"

        parts: list[str] = []
        parts.append(
            f"Sensor tipo {sensor_type or 'desconocido'} con severidad actual {severity} "
            f"y patrón {pattern}. Acción recomendada: {action}. "
        )
        parts.append(
            "Ventanas 1s/5s/10s: "
            f"media_base={analysis.baseline_mean:.4f}, "
            f"último_valor={analysis.last_value:.4f}, "
            f"z_score={analysis.z_score_last:.2f}, "
            f"tendencia_larga={trend_desc}. "
        )

        explanation = human + " " + "".join(parts)
        return severity, action, explanation

    def _should_emit_state_event(
        self,
        *,
        prev_state: Optional[SensorState],
        new_state: SensorState,
        analysis: OnlineAnalysis,
        value_within_warning_range: bool = False,
    ) -> tuple[bool, str]:
        """Decide si se debe insertar un evento de estado y qué event_code usar.
        
        REGLA DE DOMINIO:
        - Si valor dentro del rango del usuario → NO emitir eventos
        - Solo emitir para cambios SIGNIFICATIVOS de severidad
        - NO generar eventos masivos por oscilaciones normales
        - Un evento por transición, no por lectura
        """

        # =========================================================================
        # REGLA 1: Si valor dentro del rango del usuario → NO emitir eventos
        # =========================================================================
        if value_within_warning_range:
            return False, prev_state.last_event_code if prev_state else "ML_BEHAVIOR_STATE"

        # =========================================================================
        # REGLA 2: Primer evento para el sensor → NO emitir (esperar baseline)
        # =========================================================================
        if prev_state is None:
            return False, "ML_BEHAVIOR_STATE"

        # =========================================================================
        # REGLA 3: Solo emitir si hay cambio SIGNIFICATIVO de severidad
        # =========================================================================
        severity_changed = prev_state.severity != new_state.severity
        
        if not severity_changed:
            return False, prev_state.last_event_code or "ML_BEHAVIOR_STATE"

        # =========================================================================
        # REGLA 4: Determinar código de evento según el tipo de anomalía
        # =========================================================================
        if analysis.new_transient_anomaly:
            code = "TRANSIENT_ANOMALY"
        elif analysis.is_curve_anomalous:
            code = "CURVE_ANOMALY"
        elif analysis.has_microvariation:
            code = "MICRO_VARIATION"
        else:
            code = "ML_BEHAVIOR_STATE"

        # =========================================================================
        # REGLA 5: No repetir el mismo código de evento
        # =========================================================================
        if prev_state.last_event_code == code:
            return False, code

        return True, code

    def _map_severity_to_event_type(self, severity: str) -> str:
        """Mapea severidad lógica (NORMAL/WARN/CRITICAL) a event_type."""

        sev = severity.upper()
        if sev == "CRITICAL":
            return "critical"
        if sev == "WARN":
            return "warning"
        return "notice"

    def _build_title(self, event_code: str, severity: str, pattern: str) -> str:
        return f"ML online: {event_code} (estado {severity}, patrón {pattern})"

    # ------------------------------------------------------------------
    # FIX CRÍTICO: Validación de umbrales y estado operacional
    # ------------------------------------------------------------------

    def _can_sensor_emit_events(self, sensor_id: int) -> bool:
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

    def _is_value_within_warning_range(self, sensor_id: int, value: float) -> bool:
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
        # FIX: Siempre recargar umbrales desde BD (sin cache) para evitar falsos positivos
        # El cache puede tener datos obsoletos si los umbrales se configuraron después
        thresholds = self._load_warning_thresholds(sensor_id)
        
        if not thresholds:
            # Sin umbrales configurados, ML puede operar libremente
            logger.debug(
                "[ML_THRESHOLD_CHECK] sensor_id=%s value=%.4f NO_THRESHOLDS - ML puede operar",
                sensor_id, value
            )
            return False
        
        warning_min = thresholds.get("warning_min")
        warning_max = thresholds.get("warning_max")
        
        # Si no hay límites de warning, no podemos determinar si está dentro
        if warning_min is None and warning_max is None:
            logger.debug(
                "[ML_THRESHOLD_CHECK] sensor_id=%s value=%.4f NO_LIMITS - ML puede operar",
                sensor_id, value
            )
            return False
        
        # Verificar si está dentro del rango
        within_range = True
        if warning_min is not None and value < warning_min:
            within_range = False  # Fuera del rango (por debajo)
        if warning_max is not None and value > warning_max:
            within_range = False  # Fuera del rango (por arriba)
        
        # Log para debug
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

    # ------------------------------------------------------------------
    # Persistencia de eventos ML + notificaciones
    # ------------------------------------------------------------------

    def _get_device_id(self, conn: Connection, sensor_id: int) -> int:
        if sensor_id in self._device_cache:
            return self._device_cache[sensor_id]
        device_id = get_device_id_for_sensor(conn, sensor_id)
        self._device_cache[sensor_id] = device_id
        return device_id

    def _insert_ml_event(
        self,
        *,
        sensor_id: int,
        sensor_type: str,
        severity_label: str,
        event_type: str,
        event_code: str,
        title: str,
        explanation: str,
        recommended_action: str,
        analysis: OnlineAnalysis,
        ts_utc: float,
        prediction_id: Optional[int],
        extra_payload: Optional[dict],
    ) -> None:
        """Inserta un ml_event y crea una notificación asociada.

        - ml_events: registro detallado del evento ML.
        - alert_notifications: estado de notificación (read/unread).
        
        LÓGICA DE DOMINIO (FIX 2026-01-28 - Auditoría Delta Spike):
        - TODOS los eventos ML tienen cooldown de 5 min (incluyendo DELTA_SPIKE)
        - ALERT activo NO bloquea eventos ML (pueden coexistir)
        - Deduplicación para eventos activos del mismo tipo
        
        NOTA: DELTA_SPIKE ahora SÍ tiene cooldown para evitar spam de notificaciones.
        El SP ya aplica cooldown, pero el ML Service también debe respetarlo.
        """

        engine = get_engine()
        with engine.begin() as conn:  # type: ignore[call-arg]
            # =========================================================================
            # LÓGICA DE DOMINIO PARA EVENTOS ML (FIX 2026-01-28)
            # =========================================================================
            # 
            # REGLAS ACTUALIZADAS:
            # 1. TODOS los eventos ML tienen cooldown de 5 min (incluyendo DELTA_SPIKE)
            # 2. ALERT activo NO bloquea eventos ML (pueden coexistir)
            # 3. Deduplicación para eventos activos del mismo tipo
            #
            # JUSTIFICACIÓN: DELTA_SPIKE generaba spam de notificaciones porque
            # estaba exento de cooldown. El SP ya aplica cooldown, pero si el ML
            # Service genera eventos por su cuenta, también debe respetarlo.
            #
            # =========================================================================
            
            # Verificar cooldown: no generar si hay evento reciente del mismo tipo
            recent_event = conn.execute(
                text("""
                    SELECT TOP 1 1 FROM dbo.ml_events
                    WHERE sensor_id = :sensor_id
                      AND event_code = :event_code
                      AND created_at >= DATEADD(MINUTE, -5, GETDATE())
                """),
                {"sensor_id": sensor_id, "event_code": event_code}
            ).fetchone()
            
            if recent_event:
                logger.debug(
                    "[ML_COOLDOWN] sensor_id=%s event_code=%s in cooldown, skipping",
                    sensor_id, event_code
                )
                return
            
            # Verificar deduplicación: no generar si ya hay evento activo del mismo tipo
            active_same_event = conn.execute(
                text("""
                    SELECT TOP 1 1 FROM dbo.ml_events
                    WHERE sensor_id = :sensor_id
                      AND event_code = :event_code
                      AND status = 'active'
                """),
                {"sensor_id": sensor_id, "event_code": event_code}
            ).fetchone()
            
            if active_same_event:
                logger.debug(
                    "[ML_DEDUPE] sensor_id=%s already has active %s event, skipping",
                    sensor_id, event_code
                )
                return

            device_id = self._get_device_id(conn, sensor_id)

            base_payload: dict = {
                "severity": severity_label,
                "behavior_pattern": analysis.behavior_pattern,
                "recommended_action": recommended_action,
                "sensor_type": sensor_type,
                "baseline_mean": analysis.baseline_mean,
                "last_value": analysis.last_value,
                "z_score_last": analysis.z_score_last,
                "is_curve_anomalous": analysis.is_curve_anomalous,
                "has_microvariation": analysis.has_microvariation,
                "microvariation_delta": analysis.microvariation_delta,
            }
            if extra_payload:
                base_payload.update(extra_payload)

            payload_json = json.dumps(base_payload, ensure_ascii=False)

            # 1) Insertar evento ML y obtener su ID
            row = conn.execute(
                text(
                    """
                    INSERT INTO dbo.ml_events (
                      device_id,
                      sensor_id,
                      prediction_id,
                      event_type,
                      event_code,
                      title,
                      message,
                      status,
                      created_at,
                      payload
                    )
                    OUTPUT INSERTED.id
                    VALUES (
                      :device_id,
                      :sensor_id,
                      :prediction_id,
                      :event_type,
                      :event_code,
                      :title,
                      :message,
                      'active',
                      DATEADD(second, :ts_utc, '1970-01-01'),
                      :payload
                    )
                    """
                ),
                {
                    "device_id": device_id,
                    "sensor_id": sensor_id,
                    "prediction_id": prediction_id,
                    "event_type": event_type,
                    "event_code": event_code,
                    "title": title,
                    "message": explanation,
                    "ts_utc": ts_utc,
                    "payload": payload_json,
                },
            ).fetchone()

            if not row:
                return

            event_id = int(row[0])

            # 2) Crear notificación no leída (separada de ml_events)
            #
            # Tabla esperada (MVP): dbo.alert_notifications
            #   - id (identity)
            #   - source (ej: 'ml_event')
            #   - source_event_id (FK lógico a ml_events.id)
            #   - severity
            #   - title
            #   - message
            #   - is_read (bit)
            #   - created_at (datetime)
            try:
                conn.execute(
                    text(
                        """
                        INSERT INTO dbo.alert_notifications (
                          source,
                          source_event_id,
                          severity,
                          title,
                          message,
                          is_read,
                          created_at
                        )
                        VALUES (
                          :source,
                          :source_event_id,
                          :severity,
                          :title,
                          :message,
                          0,
                          GETDATE()
                        )
                        """
                    ),
                    {
                        "source": "ml_event",
                        "source_event_id": event_id,
                        "severity": event_type,
                        "title": title,
                        "message": explanation,
                    },
                )
            except Exception:
                # Si la tabla aún no existe o hay un problema de esquema,
                # no rompemos el flujo de ML; solo dejamos pasar.
                logger.exception("No se pudo insertar en alert_notifications (ML online)")

    # ------------------------------------------------------------------
    # Autovalidación de predicciones (PREDICTION_DEVIATION)
    # ------------------------------------------------------------------

    def _should_dedupe_prediction_deviation(
        self,
        conn: Connection,
        *,
        sensor_id: int,
    ) -> bool:
        cfg = self._cfg
        row = conn.execute(
            text(
                """
                SELECT TOP 1 1
                FROM dbo.ml_events
                WHERE sensor_id = :sensor_id
                  AND event_code = 'PREDICTION_DEVIATION'
                  AND status IN ('active', 'acknowledged')
                  AND created_at >= DATEADD(minute, -:mins, GETDATE())
                ORDER BY created_at DESC
                """
            ),
            {"sensor_id": sensor_id, "mins": cfg.dedupe_minutes_prediction_deviation},
        ).fetchone()
        return row is not None

    def _check_prediction_deviation(
        self,
        *,
        sensor_id: int,
        reading: Reading,
        analysis: OnlineAnalysis,
    ) -> None:
        """Verifica desviación entre predicción y valor real.
        
        REGLA DE DOMINIO CRÍTICA:
        Si el valor está dentro del rango WARNING del usuario, NO generar
        eventos de desviación. El usuario definió ese rango como "normal".
        """
        cfg = self._cfg
        
        # =========================================================================
        # FIX CRÍTICO: Si el valor está dentro del rango del usuario, NO generar
        # eventos de PREDICTION_DEVIATION. El usuario definió ese rango como normal.
        # =========================================================================
        if self._is_value_within_warning_range(sensor_id, float(reading.value)):
            return
        
        engine = get_engine()

        reading_dt = datetime.fromtimestamp(float(reading.timestamp), tz=timezone.utc)

        with engine.connect() as conn:  # type: ignore[call-arg]
            # FIX PUNTO 2.3: Eliminar CAST AS float, mantener precisión DECIMAL(15,5)
            row = conn.execute(
                text(
                    """
                    SELECT TOP 1 id, [predicted_value], target_timestamp
                    FROM dbo.predictions
                    WHERE sensor_id = :sensor_id
                      AND ABS(DATEDIFF(second, target_timestamp, :ts)) <= :tol
                    ORDER BY ABS(DATEDIFF(second, target_timestamp, :ts)) ASC
                    """
                ),
                {
                    "sensor_id": sensor_id,
                    "ts": reading_dt.replace(tzinfo=None),
                    "tol": cfg.prediction_time_tolerance_seconds,
                },
            ).fetchone()

            if not row:
                return

            prediction_id, predicted_value, _target_ts = row
            # Python Decimal → float mantiene mejor precisión que SQL CAST
            predicted_value_f = float(predicted_value) if predicted_value is not None else 0.0

            error_abs = abs(float(reading.value) - predicted_value_f)
            denom = max(abs(predicted_value_f), 1e-6)
            error_rel = error_abs / denom

            if (
                error_abs < cfg.prediction_error_absolute
                and error_rel < cfg.prediction_error_relative
            ):
                return

            if self._should_dedupe_prediction_deviation(conn, sensor_id=sensor_id):
                return

        # Si llegamos aquí, generamos el evento usando una nueva transacción
        severity_label = "WARN" if error_rel < 0.5 else "CRITICAL"
        event_type = self._map_severity_to_event_type(severity_label)

        explanation = (
            "Desviación significativa entre la predicción del modelo y el valor real. "
            f"real={float(reading.value):.4f} predicho={predicted_value_f:.4f} "
            f"error_abs={error_abs:.4f} error_rel={error_rel:.2%}."
        )
        recommended_action = (
            "Revisar la configuración del modelo y las condiciones del sensor. "
            "Si la desviación persiste, considerar recalibrar o reentrenar el modelo."
        )

        extra_payload = {
            "prediction_id": int(prediction_id),
            "predicted_value": predicted_value_f,
            "error_abs": error_abs,
            "error_rel": error_rel,
        }

        self._insert_ml_event(
            sensor_id=sensor_id,
            sensor_type=reading.sensor_type,
            severity_label=severity_label,
            event_type=event_type,
            event_code="PREDICTION_DEVIATION",
            title="ML online: PREDICTION_DEVIATION",
            explanation=explanation,
            recommended_action=recommended_action,
            analysis=analysis,
            ts_utc=float(reading.timestamp),
            prediction_id=int(prediction_id),
            extra_payload=extra_payload,
        )


def run_stream(broker: ReadingBroker) -> None:
    """Entry point del runner online.

    Recibe un `ReadingBroker` y se subscribe con un handler simple.
    """

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    logger.info("[ML-STREAM] Iniciando runner ML online (broker abstracto)")

    processor = SimpleMlOnlineProcessor()

    def handler(reading: Reading) -> None:
        processor.handle_reading(reading)

    broker.subscribe(handler)


def main() -> None:
    """Punto de entrada CLI de ejemplo usando InMemoryReadingBroker.

    Nota: esto asume que el broker se comparte en el mismo proceso.
    Para otros brokers, solo cambias la construcción aquí.
    """

    from iot_machine_learning.ml_service.in_memory_broker import InMemoryReadingBroker

    broker = InMemoryReadingBroker()
    run_stream(broker)


if __name__ == "__main__":
    main()
