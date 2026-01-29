"""Runner modular para consumo online de lecturas por ML.

REFACTORIZADO: Este archivo fue reducido de 1088 líneas a ~250 líneas
extrayendo responsabilidades a módulos separados:

- models/sensor_state.py: Estado del sensor
- models/online_analysis.py: Resultado del análisis
- services/window_analyzer.py: Análisis de ventanas deslizantes
- services/threshold_validator.py: Validación de umbrales
- services/explanation_builder.py: Construcción de explicaciones
- services/event_persister.py: Persistencia de eventos

Este módulo ahora solo contiene la orquestación del flujo principal.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from sqlalchemy import text

from iot_ingest_services.common.db import get_engine
from iot_machine_learning.ml_service.config.ml_config import (
    DEFAULT_ML_CONFIG,
    OnlineBehaviorConfig,
)
from iot_machine_learning.ml_service.reading_broker import Reading, ReadingBroker
from iot_machine_learning.ml_service.sliding_window_buffer import SlidingWindowBuffer
from iot_machine_learning.ml_service.features.ml_features import (
    MLFeatures,
    get_ml_features_producer,
)

from .models import SensorState, OnlineAnalysis
from .services import (
    WindowAnalyzer,
    ThresholdValidator,
    ExplanationBuilder,
    MLEventPersister,
)

logger = logging.getLogger(__name__)


class SimpleMlOnlineProcessor:
    """Procesador online modular usando SlidingWindowBuffer.

    - Mantiene ventanas deslizantes 1s/5s/10s por sensor.
    - Delega análisis, validación y persistencia a servicios especializados.
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
        
        # Servicios modulares
        self._window_analyzer = WindowAnalyzer(self._cfg)
        self._threshold_validator = ThresholdValidator()
        self._explanation_builder = ExplanationBuilder()
        self._event_persister = MLEventPersister()
        
        # FASE 2.2: MLFeaturesProducer para producir features SIEMPRE
        self._features_producer = get_ml_features_producer()

    def handle_reading(self, reading: Reading) -> None:
        """Procesa una lectura y emite eventos en BD cuando cambia el estado."""
        start_time = time.time()

        sensor_id = reading.sensor_id
        reading_ts = float(reading.timestamp)
        ingest_latency_ms = (start_time - reading_ts) * 1000 if reading_ts > 0 else 0
        
        if ingest_latency_ms > 1000:
            logger.warning(
                "[ML_LATENCY] High ingest latency sensor_id=%s latency_ms=%.1f",
                sensor_id, ingest_latency_ms
            )
        
        # FASE 2.2: PRODUCIR FEATURES SIEMPRE
        ml_features = self._features_producer.process_reading(
            sensor_id=sensor_id,
            value=float(reading.value),
            timestamp=reading_ts,
            sensor_type=reading.sensor_type,
        )
        self._publish_ml_features(ml_features)
        
        # Verificar estado operacional
        can_emit_events = self._threshold_validator.can_sensor_emit_events(sensor_id)
        if not can_emit_events:
            self._buffer.add_reading(
                sensor_id=sensor_id,
                value=float(reading.value),
                timestamp=reading_ts,
                windows=(1.0, 5.0, 10.0),
            )
            return
        
        # Verificar rango WARNING del usuario
        value_within_warning_range = self._threshold_validator.is_value_within_warning_range(
            sensor_id, float(reading.value)
        )
        
        # Agregar lectura al buffer y obtener estadísticas
        stats_by_window = self._buffer.add_reading(
            sensor_id=sensor_id,
            value=float(reading.value),
            timestamp=reading_ts,
            windows=(1.0, 5.0, 10.0),
        )

        if not stats_by_window:
            return

        # Analizar ventanas
        prev_state = self._last_state.get(sensor_id)
        analysis = self._window_analyzer.analyze_windows(stats_by_window, prev_state)

        # Construir explicación
        severity, action, explanation = self._explanation_builder.build_explanation(
            sensor_type=reading.sensor_type,
            analysis=analysis,
            value_within_warning_range=value_within_warning_range,
        )

        # Actualizar estado
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

        # Decidir si emitir evento
        should_emit, event_code = self._explanation_builder.should_emit_state_event(
            prev_state=prev_state,
            new_state=new_state,
            analysis=analysis,
            value_within_warning_range=value_within_warning_range,
        )

        # Validación adicional
        if should_emit and value_within_warning_range and severity in ("WARN", "CRITICAL"):
            logger.debug(
                "[ML_SUPPRESSED] sensor_id=%s value=%.4f within warning range, "
                "suppressing %s event",
                sensor_id, float(reading.value), severity
            )
            should_emit = False
            new_state.severity = "NORMAL"

        if should_emit:
            event_type = self._explanation_builder.map_severity_to_event_type(severity)
            title = self._explanation_builder.build_title(event_code, severity, analysis.behavior_pattern)

            processing_time_ms = (time.time() - start_time) * 1000
            logger.info(
                "[ML_EVENT] Emitting event sensor_id=%s code=%s severity=%s processing_ms=%.1f",
                sensor_id, event_code, severity, processing_time_ms
            )

            self._event_persister.insert_ml_event(
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

        self._last_state[sensor_id] = new_state

        # Autovalidación de predicción
        try:
            self._check_prediction_deviation(sensor_id=sensor_id, reading=reading, analysis=analysis)
        except Exception:
            logger.exception(
                "Error evaluando desviación de predicción para sensor_id=%s", sensor_id
            )

    def _publish_ml_features(self, features: MLFeatures) -> None:
        """Publica ML features a telemetría para visualización."""
        try:
            logger.info(
                "[ML_FEATURES] sensor_id=%s baseline=%.4f deviation=%.4f "
                "z_score=%.4f confidence=%.4f pattern=%s anomaly=%s",
                features.sensor_id,
                features.baseline,
                features.deviation,
                features.z_score,
                features.confidence,
                features.pattern_detected,
                features.is_anomalous,
            )
        except Exception as e:
            logger.warning(
                "[ML_FEATURES_PUBLISH_ERROR] sensor_id=%s error=%s",
                features.sensor_id, str(e)
            )

    def _check_prediction_deviation(
        self,
        *,
        sensor_id: int,
        reading: Reading,
        analysis: OnlineAnalysis,
    ) -> None:
        """Verifica desviación entre predicción y valor real."""
        cfg = self._cfg
        
        if self._threshold_validator.is_value_within_warning_range(sensor_id, float(reading.value)):
            return
        
        engine = get_engine()
        reading_dt = datetime.fromtimestamp(float(reading.timestamp), tz=timezone.utc)

        with engine.connect() as conn:
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
            predicted_value_f = float(predicted_value) if predicted_value is not None else 0.0

            error_abs = abs(float(reading.value) - predicted_value_f)
            denom = max(abs(predicted_value_f), 1e-6)
            error_rel = error_abs / denom

            if (
                error_abs < cfg.prediction_error_absolute
                and error_rel < cfg.prediction_error_relative
            ):
                return

            if self._event_persister.should_dedupe_prediction_deviation(
                conn, sensor_id=sensor_id, dedupe_minutes=cfg.dedupe_minutes_prediction_deviation
            ):
                return

        severity_label = "WARN" if error_rel < 0.5 else "CRITICAL"
        event_type = self._explanation_builder.map_severity_to_event_type(severity_label)

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

        self._event_persister.insert_ml_event(
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
    """Entry point del runner online."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    logger.info("[ML-STREAM] Iniciando runner ML online (modular)")

    processor = SimpleMlOnlineProcessor()

    def handler(reading: Reading) -> None:
        processor.handle_reading(reading)

    broker.subscribe(handler)


def main() -> None:
    """Punto de entrada CLI usando InMemoryReadingBroker."""
    from iot_machine_learning.ml_service.in_memory_broker import InMemoryReadingBroker

    broker = InMemoryReadingBroker()
    run_stream(broker)


if __name__ == "__main__":
    main()
