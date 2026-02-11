"""Servicio de dominio para detección de anomalías.

Orquesta detectores de anomalías usando ports.  Soporta voting
entre múltiples detectores para reducir falsos positivos.

Responsabilidades:
- Ejecutar detectores configurados.
- Combinar votos (si hay múltiples detectores).
- Generar AnomalyResult del dominio.
- Delegar auditoría al AuditPort.
"""

from __future__ import annotations

import logging
import uuid
from typing import List, Optional

from ..entities.anomaly import AnomalyResult, AnomalySeverity
from ..entities.sensor_reading import SensorWindow
from ..ports.anomaly_detection_port import AnomalyDetectionPort
from ..ports.audit_port import AuditPort

logger = logging.getLogger(__name__)

# Umbral de voting para declarar anomalía (score promedio > threshold)
_DEFAULT_VOTING_THRESHOLD: float = 0.5


class AnomalyDomainService:
    """Servicio de dominio que orquesta detección de anomalías.

    Soporta un solo detector o un ensemble con voting.

    Attributes:
        _detectors: Lista de detectores disponibles.
        _voting_threshold: Umbral para declarar anomalía por voting.
        _audit: Port de auditoría (opcional).
    """

    def __init__(
        self,
        detectors: List[AnomalyDetectionPort],
        voting_threshold: float = _DEFAULT_VOTING_THRESHOLD,
        audit: Optional[AuditPort] = None,
    ) -> None:
        if not detectors:
            raise ValueError("Se requiere al menos un detector de anomalías")

        self._detectors = detectors
        self._voting_threshold = voting_threshold
        self._audit = audit

    def detect(self, window: SensorWindow) -> AnomalyResult:
        """Detecta anomalías combinando votos de todos los detectores.

        Args:
            window: Ventana temporal del sensor.

        Returns:
            ``AnomalyResult`` con score combinado y votos individuales.
        """
        if window.is_empty:
            return AnomalyResult.normal(series_id=str(window.sensor_id))

        trace_id = str(uuid.uuid4())[:12]
        votes: dict[str, float] = {}
        explanations: list[str] = []

        for detector in self._detectors:
            try:
                if not detector.is_trained():
                    logger.debug(
                        "detector_not_trained_skip",
                        extra={"detector": detector.name, "series_id": str(window.sensor_id)},
                    )
                    continue

                result = detector.detect(window)
                votes[detector.name] = result.score

                if result.is_anomaly and result.explanation:
                    explanations.append(f"[{detector.name}] {result.explanation}")

            except Exception as exc:
                logger.warning(
                    "detector_failed",
                    extra={
                        "detector": detector.name,
                        "series_id": str(window.sensor_id),
                        "error": str(exc),
                    },
                )
                # Detector que falla no vota (fail-open)
                continue

        # Sin votos → normal
        if not votes:
            return AnomalyResult.normal(series_id=str(window.sensor_id))

        # Combinar votos (promedio ponderado uniforme)
        avg_score = sum(votes.values()) / len(votes)
        is_anomaly = avg_score > self._voting_threshold

        # Confianza basada en consenso entre detectores
        if len(votes) > 1:
            scores = list(votes.values())
            mean_s = sum(scores) / len(scores)
            variance = sum((s - mean_s) ** 2 for s in scores) / len(scores)
            std_s = variance ** 0.5
            confidence = max(0.5, 1.0 - std_s)
        else:
            confidence = 0.7  # Un solo detector → confianza moderada

        severity = AnomalySeverity.from_score(avg_score)
        explanation = " | ".join(explanations) if explanations else "Valor normal"

        anomaly_result = AnomalyResult(
            series_id=str(window.sensor_id),
            is_anomaly=is_anomaly,
            score=avg_score,
            method_votes=votes,
            confidence=confidence,
            explanation=explanation,
            severity=severity,
            audit_trace_id=trace_id,
        )

        # Auditoría
        if self._audit is not None and is_anomaly:
            try:
                self._audit.log_anomaly(
                    sensor_id=window.sensor_id,  # AuditPort still uses sensor_id
                    value=window.last_value or 0.0,
                    score=avg_score,
                    explanation=explanation,
                    trace_id=trace_id,
                )
            except Exception:
                logger.warning("audit_log_failed", extra={"trace_id": trace_id})

        logger.info(
            "anomaly_detection_complete",
            extra={
                "series_id": str(window.sensor_id),
                "is_anomaly": is_anomaly,
                "score": round(avg_score, 4),
                "votes": {k: round(v, 4) for k, v in votes.items()},
                "severity": severity.value,
                "trace_id": trace_id,
            },
        )

        return anomaly_result

    def train_all(self, historical_values: List[float]) -> None:
        """Entrena todos los detectores con datos históricos.

        Args:
            historical_values: Serie temporal de entrenamiento.
        """
        for detector in self._detectors:
            try:
                detector.train(historical_values)
                logger.info(
                    "detector_trained",
                    extra={"detector": detector.name, "n_points": len(historical_values)},
                )
            except Exception as exc:
                logger.error(
                    "detector_training_failed",
                    extra={"detector": detector.name, "error": str(exc)},
                )
