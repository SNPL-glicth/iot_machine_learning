"""Ciclo de vida de una predicción (FASE 3).

Contract v1 — el único camino que alimenta el aprendizaje posterior es:

    PENDING -> ACTIVE -> WAITING_OUTCOME -> EVALUATED -> REWARDED -> ARCHIVED

Reglas de negocio:
    * ``WAITING_OUTCOME`` NO produce reward (falta el desenlace).
    * ``INVALIDATED`` NO produce reward (descartada).
    * ``PENDING``/``ACTIVE`` NO producen reward (no hay evaluación).
    * Solo ``EVALUATED -> REWARDED`` materializa el reward.

Estado terminales: ``ARCHIVED`` e ``INVALIDATED``.
"""

from __future__ import annotations

import math
from enum import Enum


class PredictionStatus(Enum):
    """Estados del ciclo de vida de una predicción."""

    PENDING = "pending"
    ACTIVE = "active"
    WAITING_OUTCOME = "waiting_outcome"
    EVALUATED = "evaluated"
    REWARDED = "rewarded"
    ARCHIVED = "archived"
    INVALIDATED = "invalidated"


_ALLOWED_TRANSITIONS: dict[PredictionStatus, frozenset[PredictionStatus]] = {
    PredictionStatus.PENDING: frozenset(
        {PredictionStatus.ACTIVE, PredictionStatus.INVALIDATED, PredictionStatus.ARCHIVED}
    ),
    PredictionStatus.ACTIVE: frozenset(
        {
            PredictionStatus.WAITING_OUTCOME,
            PredictionStatus.INVALIDATED,
            PredictionStatus.ARCHIVED,
        }
    ),
    PredictionStatus.WAITING_OUTCOME: frozenset(
        {PredictionStatus.EVALUATED, PredictionStatus.INVALIDATED, PredictionStatus.ARCHIVED}
    ),
    PredictionStatus.EVALUATED: frozenset(
        {PredictionStatus.REWARDED, PredictionStatus.ARCHIVED}
    ),
    PredictionStatus.REWARDED: frozenset({PredictionStatus.ARCHIVED}),
    PredictionStatus.ARCHIVED: frozenset(),
    PredictionStatus.INVALIDATED: frozenset(),
}


class InvalidTransitionError(ValueError):
    """Transición de estado no permitida por el contrato v1."""


def validate_transition(
    current: PredictionStatus, next_status: PredictionStatus
) -> None:
    """Valida la máquina de estados de la predicción.

    Raises:
        InvalidTransitionError: si ``current -> next_status`` no está permitida.
    """
    if not isinstance(current, PredictionStatus) or not isinstance(
        next_status, PredictionStatus
    ):
        raise InvalidTransitionError(
            f"estados inválidos: {current!r} -> {next_status!r}"
        )
    if next_status not in _ALLOWED_TRANSITIONS[current]:
        raise InvalidTransitionError(
            f"transición no permitida: {current.value} -> {next_status.value}"
        )


def is_terminal(status: PredictionStatus) -> bool:
    """``True`` si el estado es terminal (no admite transiciones)."""
    return not _ALLOWED_TRANSITIONS[status]


def validate_state_consistency(
    *,
    status: PredictionStatus,
    outcome: object | None,
    evaluation: object | None,
    reward: object | None,
    created_at: float,
    updated_at: float,
) -> None:
    """Valida campos inline coherentes con el estado (estados imposibles).

    Función pura: no conoce la entidad ``Prediction``, solo sus campos.

    ARCHIVED conserva su historia (outcome/evaluation/reward) para
    diagnóstico; INVALIDATED conserva outcome si ya había llegado, pero
    jamás evaluation ni reward (nunca se recompensa).
    """
    if not isinstance(status, PredictionStatus):
        raise TypeError(f"status inválido: {status!r}")
    if outcome is not None and status not in {
        PredictionStatus.WAITING_OUTCOME,
        PredictionStatus.EVALUATED,
        PredictionStatus.REWARDED,
        PredictionStatus.ARCHIVED,
        PredictionStatus.INVALIDATED,
    }:
        raise ValueError(f"outcome no puede existir en estado {status.value}")
    if evaluation is not None and status not in {
        PredictionStatus.EVALUATED,
        PredictionStatus.REWARDED,
        PredictionStatus.ARCHIVED,
    }:
        raise ValueError(f"evaluation no puede existir en estado {status.value}")
    if reward is not None and status not in {
        PredictionStatus.REWARDED,
        PredictionStatus.ARCHIVED,
    }:
        raise ValueError(f"reward no puede existir en estado {status.value}")
    if status in {
        PredictionStatus.WAITING_OUTCOME,
        PredictionStatus.EVALUATED,
        PredictionStatus.REWARDED,
    } and outcome is None:
        raise ValueError(f"estado {status.value} exige outcome")
    if status in {PredictionStatus.EVALUATED, PredictionStatus.REWARDED} and (
        evaluation is None
    ):
        raise ValueError(f"estado {status.value} exige evaluation")
    if status == PredictionStatus.REWARDED and reward is None:
        raise ValueError("estado rewarded exige reward")
    if not math.isfinite(created_at) or not math.isfinite(updated_at):
        raise ValueError("created_at/updated_at inválidos")
    if updated_at < created_at:
        raise ValueError(
            f"updated_at no puede ser anterior a created_at: "
            f"{updated_at} < {created_at}"
        )
