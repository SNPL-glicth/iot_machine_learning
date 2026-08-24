"""Prediction Transitions (FASE 3)."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from .lifecycle import (
    InvalidTransitionError,
    PredictionStatus,
    validate_state_consistency,
    validate_transition,
)
from .prediction_entity import Prediction


def transition(self: Prediction, next_status: PredictionStatus, **fields: Any) -> Prediction:
    """Crea una nueva Prediction con estado actualizado (inmutable)."""
    validate_transition(self.status, next_status)
    if "updated_at" not in fields:
        fields["updated_at"] = time.time()
    updated = replace(self, status=next_status, **fields)
    validate_state_consistency(
        status=updated.status,
        outcome=updated.outcome,
        evaluation=updated.evaluation,
        reward=updated.reward,
        created_at=updated.created_at,
        updated_at=updated.updated_at,
    )
    return updated


def activate(self: Prediction) -> Prediction:
    """PENDING -> ACTIVE."""
    return transition(self, PredictionStatus.ACTIVE)


def to_waiting_outcome(self: Prediction, outcome: "Outcome") -> Prediction:
    """ACTIVE -> WAITING_OUTCOME, vinculando el Outcome del horizonte."""
    from .prediction_guards import guard_outcome
    guard_outcome(self, outcome)
    return transition(self, PredictionStatus.WAITING_OUTCOME, outcome=outcome)


def evaluate(self: Prediction, outcome: "Outcome") -> Prediction:
    """WAITING_OUTCOME -> EVALUATED, calculando la evaluación pura."""
    from .evaluation import evaluate_prediction
    from .prediction_guards import guard_outcome

    guard_outcome(self, outcome)
    evaluation = evaluate_prediction(self, outcome)
    return transition(
        self, PredictionStatus.EVALUATED, evaluation=evaluation
    )


def issue_reward(self: Prediction, config: "RewardConfig") -> Prediction:
    """EVALUATED -> REWARDED: única vía para materializar un reward."""
    from .reward import RewardConfig, compute_reward

    if not isinstance(config, RewardConfig):
        raise TypeError("config debe ser RewardConfig")
    if self.status != PredictionStatus.EVALUATED:
        raise InvalidTransitionError(
            f"solo EVALUATED puede recibir reward (estado: {self.status.value})"
        )
    if self.outcome is None or self.evaluation is None:
        raise ValueError("EVALUATED sin outcome/evaluation: estado inconsistente")
    reward = compute_reward(self, self.outcome, self.evaluation, config)
    return transition(self, PredictionStatus.REWARDED, reward=reward)


def invalidate(self: Prediction, reason: str | None = None) -> Prediction:
    """Descarta la predicción (NUNCA produce reward).

    Args:
        reason: Razón de invalidación (ej. "provider_gap" para FASE 6).
    """
    return transition(self, PredictionStatus.INVALIDATED, invalidation_reason=reason)


def archive(self: Prediction) -> Prediction:
    """Archiva la predicción (post-reward o sin reward)."""
    return transition(self, PredictionStatus.ARCHIVED)


def can_produce_reward(self: Prediction) -> bool:
    """``True`` solo si el estado admite llegar a REWARDED."""
    try:
        validate_transition(self.status, PredictionStatus.REWARDED)
        return True
    except InvalidTransitionError:
        return False