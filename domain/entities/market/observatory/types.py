"""Observatory dataclasses (FASE 10)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ObservationRow:
    """Contrato de fila del observatorio (corte de ``market_predictions``)."""

    prediction_id: str
    emitted_at: float
    strategy: str
    horizon_seconds: int
    regime: str | None
    probability_up: float
    direction_correct: bool | None
    outcome_return_realized: float | None
    reward_total: float | None
    calibration_error: float | None
    status: str
    data_status: str | None


@dataclass(frozen=True, slots=True)
class ObservatorySummary:
    """Conteos del ciclo de vida + métricas globales de las evaluadas."""

    total: int
    evaluated: int
    pending: int
    invalidated: int
    archived: int
    stale: int
    hits: int
    accuracy: float
    wilson_lb: float
    mean_reward: float
    mean_calibration_error: float


@dataclass(frozen=True, slots=True)
class DimensionStat:
    """Una fila de BY HORIZON / BY STRATEGY / BY REGIME."""

    label: str
    predictions: int  # predicciones totales del grupo (incl. sin resolver)
    n: int  # evaluadas (rewarded con outcome)
    hits: int
    accuracy: float
    wilson_lb: float
    mean_reward: float


@dataclass(frozen=True, slots=True)
class LearningPoint:
    """Accuracy acumulada en las primeras ``n`` observaciones (por tiempo)."""

    n: int
    accuracy: float
    wilson_lb: float


@dataclass(frozen=True, slots=True)
class ContextLearning:
    """Learning curve + requerimiento de evidencia de un contexto."""

    label: str
    points: tuple[LearningPoint, ...]
    requirement: int | None


@dataclass(frozen=True, slots=True)
class BandStat:
    """Una banda de recencia (de la más antigua a la más reciente)."""

    band: int
    n: int
    hits: int
    accuracy: float
    wilson_lb: float
    mean_reward: float