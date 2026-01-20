from __future__ import annotations

from dataclasses import dataclass

from iot_machine_learning.ml_service.models.regression_model import Trend


@dataclass(frozen=True)
class PredictionExplanation:
    sensor_id: int
    predicted_value: float
    trend: Trend
    anomaly: bool
    anomaly_score: float
    confidence: float
    explanation: str

    # Campos contextuales derivados
    # risk_level: nivel de riesgo físico por umbral (LOW | MEDIUM | HIGH | NONE)
    risk_level: str
    # severity: combinación de anomalía + riesgo (info | warning | critical)
    severity: str
    action_required: bool
    recommended_action: str

    def to_dict(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "predicted_value": self.predicted_value,
            "trend": self.trend,
            "anomaly": self.anomaly,
            "anomaly_score": self.anomaly_score,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "risk_level": self.risk_level,
            "severity": self.severity,
            "action_required": self.action_required,
            "recommended_action": self.recommended_action,
        }


def build_explanation_text(
    *,
    trend: Trend,
    predicted_value: float,
    anomaly: bool,
    anomaly_score: float,
    confidence: float,
    horizon_minutes: int,
) -> str:
    base = (
        f"En los últimos minutos, los valores de este sensor muestran una tendencia '{trend}'. "
        f"El modelo estima que en aproximadamente {horizon_minutes} minutos el valor será "
        f"de alrededor de {predicted_value:.2f}. "
    )

    if anomaly:
        base += (
            "Esta predicción se considera inusual comparada con el comportamiento histórico "
            f"del sensor (anomaly_score={anomaly_score:.3f}). "
            "Se recomienda revisar el equipo o el entorno del sensor para descartar fallas "
            "o cambios bruscos inesperados."
        )
    else:
        base += (
            "La predicción está dentro de los rangos habituales observados para este sensor, "
            "por lo que no se detectan anomalías relevantes en este momento."
        )

    base += f" (confianza del modelo: {confidence:.2f} en una escala de 0 a 1)."
    return base
