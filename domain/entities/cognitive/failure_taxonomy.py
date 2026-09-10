"""Taxonomía de fallos y diagnóstico causal para evaluación post-mortem."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class FailureReason(str, Enum):
    """Categorías causales de fallos en el proceso de inferencia."""

    NONE = "none"
    REGIME_DISRUPTION = "regime_disruption"
    INERTIA_COLLAPSE = "inertia_collapse"
    ARBITER_OVERCONFIDENCE = "arbiter_overconfidence"
    ENTROPY_UNDERESTIMATION = "entropy_underestimation"
    CORRELATED_EXPERT_FAILURE = "correlated_expert_failure"


@dataclass(frozen=True)
class FailureDiagnostic:
    """Diagnóstico detallado de la causa de un fallo o acierto."""

    reason: FailureReason
    description: str
    severity: float  # [0.0, 1.0]
    expected_direction: int  # -1, 0, 1
    actual_direction: int  # -1, 0, 1
    expert_variance: float
    confidence_at_time: float
    entropy_lambda: float
    details: Dict[str, Any]

    @property
    def is_failure(self) -> bool:
        return self.reason != FailureReason.NONE


def diagnose_prediction_failure(
    predicted_val: float,
    actual_val: float,
    prev_val: float,
    confidence: float,
    lambda_t: float,
    expert_variance: float,
    is_outlier: bool = False,
    variance_threshold: float = 0.25,
    error_tolerance: float = 1e-4,
) -> FailureDiagnostic:
    """Diagnostica determinísticamente la causa raíz del error de predicción.

    Analiza la discrepancia entre el valor predicho y el valor real considerando
    el régimen estocástico, la varianza del jurado y la entropía al momento de emisión.
    """
    pred_delta = predicted_val - prev_val
    actual_delta = actual_val - prev_val

    pred_dir = 1 if pred_delta > error_tolerance else (-1 if pred_delta < -error_tolerance else 0)
    actual_dir = 1 if actual_delta > error_tolerance else (-1 if actual_delta < -error_tolerance else 0)

    # 1. ¿Fue un acierto direccional y de magnitud razonable?
    direction_correct = (pred_dir == actual_dir) and pred_dir != 0
    abs_err = abs(actual_val - predicted_val)

    if direction_correct and abs_err <= max(abs(pred_delta) * 1.5, error_tolerance):
        return FailureDiagnostic(
            reason=FailureReason.NONE,
            description="Predicción correcta y consistente con el movimiento",
            severity=0.0,
            expected_direction=pred_dir,
            actual_direction=actual_dir,
            expert_variance=expert_variance,
            confidence_at_time=confidence,
            entropy_lambda=lambda_t,
            details={"abs_error": abs_err},
        )

    # 2. Si hubo outlier o disrupción violenta detectada
    if is_outlier:
        return FailureDiagnostic(
            reason=FailureReason.REGIME_DISRUPTION,
            description="Outlier o salto anómalo no estacionario (Mahalanobis)",
            severity=1.0,
            expected_direction=pred_dir,
            actual_direction=actual_dir,
            expert_variance=expert_variance,
            confidence_at_time=confidence,
            entropy_lambda=lambda_t,
            details={"abs_error": abs_err, "is_outlier": True},
        )

    # 3. Ruptura de inercia: el modelo asumió velocidad continua pero hubo reversión opuesta
    if pred_dir != 0 and actual_dir != 0 and pred_dir != actual_dir:
        if confidence > 0.5 and expert_variance < variance_threshold:
            return FailureDiagnostic(
                reason=FailureReason.INERTIA_COLLAPSE,
                description="Colapso de inercia cinemática: reversión direccional abrupta",
                severity=min(1.0, abs_err / (abs(pred_delta) + error_tolerance)),
                expected_direction=pred_dir,
                actual_direction=actual_dir,
                expert_variance=expert_variance,
                confidence_at_time=confidence,
                entropy_lambda=lambda_t,
                details={"abs_error": abs_err},
            )

    # 4. Exceso de confianza del árbitro: varianza alta entre expertos pero confianza alta emitida
    if expert_variance >= variance_threshold and confidence > 0.6:
        return FailureDiagnostic(
            reason=FailureReason.ARBITER_OVERCONFIDENCE,
            description="El árbitro aprobó la acción a pesar de alta varianza entre expertos",
            severity=0.8,
            expected_direction=pred_dir,
            actual_direction=actual_dir,
            expert_variance=expert_variance,
            confidence_at_time=confidence,
            entropy_lambda=lambda_t,
            details={"abs_error": abs_err, "variance": expert_variance},
        )

    # 5. Subestimación de entropía: lambda_t bajo en entorno desconocido/ruidoso
    if lambda_t < 0.3 and abs_err > abs(pred_delta) * 2.0:
        return FailureDiagnostic(
            reason=FailureReason.ENTROPY_UNDERESTIMATION,
            description="Incertidumbre epistémica subestimada (lambda_t demasiado bajo)",
            severity=0.7,
            expected_direction=pred_dir,
            actual_direction=actual_dir,
            expert_variance=expert_variance,
            confidence_at_time=confidence,
            entropy_lambda=lambda_t,
            details={"abs_error": abs_err},
        )

    # 6. Fallo correlacionado general
    return FailureDiagnostic(
        reason=FailureReason.CORRELATED_EXPERT_FAILURE,
        description="Fallo de aproximación general en el jurado",
        severity=min(1.0, abs_err / (abs(actual_val) + error_tolerance)),
        expected_direction=pred_dir,
        actual_direction=actual_dir,
        expert_variance=expert_variance,
        confidence_at_time=confidence,
        entropy_lambda=lambda_t,
        details={"abs_error": abs_err},
    )
