"""Narrador de predicciones — capa Narrative.

Extraído de sensor_processor.py._build_explanation().
Responsabilidad ÚNICA: dado un resultado de predicción y metadata del sensor,
construir una explicación legible para humanos (PredictionExplanation).

No contiene I/O (no lee BD).
No calcula predicciones (eso es Modeling).
No decide severidad (eso es Reasoning — delega a SeverityClassifier).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from iot_machine_learning.ml_service.explain.explanation_builder import PredictionExplanation
    from iot_machine_learning.ml_service.repository.sensor_repository import SensorMetadata
    from iot_machine_learning.ml_service.runners.common.severity_classifier import (
        SeverityClassifier,
        SeverityResult,
    )

logger = logging.getLogger(__name__)


def build_short_message(
    severity: str,
    sensor_type: str,
    location: str,
) -> str:
    """Genera mensaje corto según severidad.

    Lógica de narrativa pura — sin I/O.

    Args:
        severity: Nivel de severidad (info | warning | critical).
        sensor_type: Tipo de sensor.
        location: Ubicación del sensor.

    Returns:
        Mensaje corto legible.
    """
    if severity == "critical":
        return f"Riesgo crítico previsto en {sensor_type} en {location}."
    if severity == "warning":
        return f"Comportamiento inusual previsto en {sensor_type} en {location}."
    return f"Predicción estable para {sensor_type} en {location}."


def build_explanation_payload(
    *,
    severity: str,
    short_message: str,
    recommended_action: str,
    predicted_value: float,
    trend: str,
    anomaly_score: float,
    confidence: float,
    horizon_minutes: int,
    risk_level: str,
    sensor_type: str,
    location: str,
) -> str:
    """Construye el JSON de explicación.

    Lógica de serialización pura — sin I/O.

    Returns:
        JSON string con la explicación completa.
    """
    payload = {
        "source": "ml_baseline",
        "severity": severity.upper(),
        "short_message": short_message,
        "recommended_action": recommended_action,
        "details": {
            "predicted_value": float(predicted_value),
            "trend": trend,
            "anomaly_score": float(anomaly_score),
            "confidence": float(confidence),
            "horizon_minutes": int(horizon_minutes),
            "risk_level": risk_level,
            "sensor_type": sensor_type,
            "location": location,
        },
    }
    return json.dumps(payload, ensure_ascii=False)


class PredictionNarrator:
    """Construye explicaciones legibles para predicciones.

    Responsabilidad ÚNICA: Narrative (generar texto humano).
    Delega clasificación de severidad a SeverityClassifier.
    No lee BD. No calcula predicciones.

    Attributes:
        _severity_classifier: Clasificador de severidad (inyectado).
    """

    def __init__(self, severity_classifier: "SeverityClassifier") -> None:
        self._severity_classifier = severity_classifier

    def build_explanation(
        self,
        *,
        sensor_meta: "SensorMetadata",
        predicted_value: float,
        trend: str,
        anomaly: bool,
        anomaly_score: float,
        confidence: float,
        horizon_minutes: int,
        user_defined_range: Optional[tuple[float, float]] = None,
    ) -> "PredictionExplanation":
        """Construye PredictionExplanation completa.

        Args:
            sensor_meta: Metadata del sensor.
            predicted_value: Valor predicho.
            trend: Tendencia (up | down | stable).
            anomaly: Si es anomalía.
            anomaly_score: Score de anomalía.
            confidence: Confianza del modelo.
            horizon_minutes: Horizonte de predicción.
            user_defined_range: Rango definido por el usuario.

        Returns:
            ``PredictionExplanation`` con todos los campos.
        """
        from iot_machine_learning.ml_service.explain.explanation_builder import (
            PredictionExplanation,
        )

        # 1. Clasificar severidad (delega a Reasoning)
        result = self._severity_classifier.classify(
            sensor_type=sensor_meta.sensor_type,
            location=sensor_meta.location,
            predicted_value=predicted_value,
            trend=trend,
            anomaly=anomaly,
            anomaly_score=anomaly_score,
            confidence=confidence,
            horizon_minutes=horizon_minutes,
            user_defined_range=user_defined_range,
        )

        # 2. Evitar contradicciones (regla de narrativa)
        effective_anomaly_score = anomaly_score
        if result.severity == "critical" and anomaly_score <= 0:
            effective_anomaly_score = 0.5

        # 3. Generar mensaje corto (narrativa pura)
        short_message = build_short_message(
            result.severity,
            sensor_meta.sensor_type,
            sensor_meta.location,
        )

        # 4. Construir JSON de explicación (serialización pura)
        explanation_json = build_explanation_payload(
            severity=result.severity,
            short_message=short_message,
            recommended_action=result.recommended_action,
            predicted_value=predicted_value,
            trend=trend,
            anomaly_score=effective_anomaly_score,
            confidence=confidence,
            horizon_minutes=horizon_minutes,
            risk_level=result.risk_level,
            sensor_type=sensor_meta.sensor_type,
            location=sensor_meta.location,
        )

        return PredictionExplanation(
            sensor_id=sensor_meta.sensor_id,
            predicted_value=predicted_value,
            trend=trend,
            anomaly=anomaly,
            anomaly_score=effective_anomaly_score,
            confidence=confidence,
            explanation=explanation_json,
            risk_level=result.risk_level,
            severity=result.severity,
            action_required=result.action_required,
            recommended_action=result.recommended_action,
        )
