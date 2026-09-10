"""Puente evaluación → atribución (FASE 7).

Convierte una Prediction evaluada (con su Outcome y su Evaluation del
ciclo) en los primitivos de ``attribute_error``. El neto realizado usa
el costo total del modelo (realizado − costos, como el meta-learner).
La degradación del dato la informa el caller (Fase 0/3: stale,
UNVERIFIED, crossed, gaps); aquí no se adivina.
"""

from __future__ import annotations

from ..costs import CostModel
from ..prediction.evaluation import Evaluation
from ..prediction.outcome import Outcome
from ..prediction.prediction import Prediction
from .error_taxonomy import ErrorAttribution, attribute_error

__all__ = ["attribute_prediction"]


def attribute_prediction(
    prediction: Prediction,
    outcome: Outcome,
    evaluation: Evaluation,
    *,
    regime_at_resolve: str | None = None,
    data_degraded: bool = False,
    cost_model: CostModel | None = None,
) -> ErrorAttribution:
    """Atribuye el error de una predicción evaluada (puro)."""
    if not isinstance(prediction, Prediction):
        raise TypeError("prediction debe ser Prediction")
    if not isinstance(outcome, Outcome):
        raise TypeError("outcome debe ser Outcome")
    if not isinstance(evaluation, Evaluation):
        raise TypeError("evaluation debe ser Evaluation")
    if cost_model is not None and not isinstance(cost_model, CostModel):
        raise TypeError("cost_model debe ser CostModel")

    regime_at_predict = (
        prediction.regime.value if prediction.regime is not None else None
    )
    tail_breach = (
        evaluation.distribution is not None
        and evaluation.distribution.tail_breach
    )
    cost = cost_model.total() if cost_model is not None else 0.0
    return attribute_error(
        direction_correct=evaluation.direction_correct,
        magnitude_error=evaluation.magnitude_error,
        calibration_error=evaluation.calibration_error,
        confidence=prediction.confidence,
        net_return=outcome.return_realized - cost,
        tail_breach=tail_breach,
        regime_at_predict=regime_at_predict,
        regime_at_resolve=regime_at_resolve,
        data_degraded=data_degraded,
    )
