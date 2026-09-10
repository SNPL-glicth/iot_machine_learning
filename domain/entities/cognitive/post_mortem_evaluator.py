"""Evaluador post-mortem para asignación de crédito y evaluación posterior."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .failure_taxonomy import FailureDiagnostic, diagnose_prediction_failure


@dataclass(frozen=True)
class PendingPrediction:
    """Predicción registrada a la espera de maduración temporal."""

    prediction_id: str
    step_created: int
    prev_value: float
    predicted_value: float
    expert_predictions: Dict[str, float]
    regime: str
    confidence: float
    lambda_t: float
    expert_variance: float
    horizon_steps: int = 11


@dataclass(frozen=True)
class PostMortemRecord:
    """Resultado auditado de la evaluación posterior una vez madurado el outcome."""

    prediction_id: str
    regime: str
    actual_value: float
    diagnostic: FailureDiagnostic
    expert_errors: Dict[str, float]
    winning_expert: str
    directional_correctness: Dict[str, bool]


class PostMortemEvaluator:
    """Manejador circular de predicciones pendientes y auditoría de resultados."""

    def __init__(self, max_pending: int = 200, max_history: int = 500) -> None:
        self._pending: deque[PendingPrediction] = deque(maxlen=max_pending)
        self._history: deque[PostMortemRecord] = deque(maxlen=max_history)
        self._current_step: int = 0

    @property
    def history(self) -> List[PostMortemRecord]:
        return list(self._history)

    def record_prediction(
        self,
        prediction_id: str,
        prev_value: float,
        predicted_value: float,
        expert_predictions: Dict[str, float],
        regime: str = "default",
        confidence: float = 0.5,
        lambda_t: float = 0.0,
        expert_variance: float = 0.0,
        horizon_steps: int = 11,
    ) -> None:
        """Registra una predicción emitida para evaluación futura."""
        self._pending.append(
            PendingPrediction(
                prediction_id=prediction_id,
                step_created=self._current_step,
                prev_value=prev_value,
                predicted_value=predicted_value,
                expert_predictions=dict(expert_predictions),
                regime=regime,
                confidence=confidence,
                lambda_t=lambda_t,
                expert_variance=expert_variance,
                horizon_steps=horizon_steps,
            )
        )

    def step(self, current_value: float, is_outlier: bool = False) -> List[PostMortemRecord]:
        """Avanza un paso de tiempo y evalúa las predicciones que hayan madurado."""
        self._current_step += 1
        evaluated_records: List[PostMortemRecord] = []
        remaining: deque[PendingPrediction] = deque(maxlen=self._pending.maxlen)

        for pending in self._pending:
            if self._current_step - pending.step_created >= pending.horizon_steps:
                record = self._evaluate_single(pending, current_value, is_outlier)
                evaluated_records.append(record)
                self._history.append(record)
            else:
                remaining.append(pending)

        self._pending = remaining
        return evaluated_records

    def _evaluate_single(
        self, pending: PendingPrediction, actual_val: float, is_outlier: bool
    ) -> PostMortemRecord:
        """Audita el desempeño del árbitro y de cada experto individual."""
        diagnostic = diagnose_prediction_failure(
            predicted_val=pending.predicted_value,
            actual_val=actual_val,
            prev_val=pending.prev_value,
            confidence=pending.confidence,
            lambda_t=pending.lambda_t,
            expert_variance=pending.expert_variance,
            is_outlier=is_outlier,
        )

        actual_delta = actual_val - pending.prev_value
        actual_dir = 1 if actual_delta > 1e-4 else (-1 if actual_delta < -1e-4 else 0)

        expert_errors: Dict[str, float] = {}
        dir_correctness: Dict[str, bool] = {}
        best_expert = "none"
        lowest_err = float("inf")

        for name, pred in pending.expert_predictions.items():
            err = abs(actual_val - pred)
            expert_errors[name] = err
            if err < lowest_err:
                lowest_err = err
                best_expert = name

            exp_delta = pred - pending.prev_value
            exp_dir = 1 if exp_delta > 1e-4 else (-1 if exp_delta < -1e-4 else 0)
            dir_correctness[name] = (exp_dir == actual_dir) and (actual_dir != 0)

        return PostMortemRecord(
            prediction_id=pending.prediction_id,
            regime=pending.regime,
            actual_value=actual_val,
            diagnostic=diagnostic,
            expert_errors=expert_errors,
            winning_expert=best_expert,
            directional_correctness=dir_correctness,
        )
