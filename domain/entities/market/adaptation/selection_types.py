"""FASE 9.4 — Selection Types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Final

from iot_machine_learning.domain.entities.market.adaptation.expert_scores import ExpertScore
from iot_machine_learning.domain.entities.market.costs import CostModel

__all__ = [
    "SelectionMode",
    "SelectionConfig",
    "ExpertNetScore",
    "SelectionResult",
    "DECISION_TRADE",
    "DECISION_HOLD",
]

DECISION_TRADE: Final = "trade"
DECISION_HOLD: Final = "hold"


class SelectionMode(Enum):
    """Modos de conversión score → pesos (FASE 9.4)."""

    SOFT = "soft"
    SELECTIVE = "selective"
    HARD_MAX = "hard_max"


@dataclass(frozen=True, slots=True, kw_only=True)
class SelectionConfig:
    """Parámetros de la selección adaptativa (todos auditables)."""

    mode: SelectionMode = SelectionMode.SOFT
    temperature: float = 1.0
    min_ratio: float = 0.5
    max_experts: int = 2
    min_n: int = 10
    min_history_days: int = 2
    min_margin: float = 0.0
    min_expected_net: float = 0.0
    risk_aversion: float = 0.1

    def __post_init__(self) -> None:
        if self.temperature <= 0.0:
            raise ValueError(f"temperature debe ser > 0: {self.temperature}")
        if not 0.0 <= self.min_ratio <= 1.0:
            raise ValueError(f"min_ratio fuera de [0, 1]: {self.min_ratio}")
        if self.max_experts < 1:
            raise ValueError(f"max_experts debe ser >= 1: {self.max_experts}")
        if self.min_n < 1:
            raise ValueError(f"min_n debe ser >= 1: {self.min_n}")
        if self.risk_aversion < 0.0:
            raise ValueError(f"risk_aversion no puede ser negativa: {self.risk_aversion}")


@dataclass(frozen=True, slots=True)
class ExpertNetScore:
    """Score neto de un experto (FASE 9.4): el edge después de pagar."""

    expert: str
    n: int
    history_days: int
    expected_return: float
    expected_cost: float
    risk_penalty: float
    expected_net: float
    calibration_quality: float
    evidence_strength: float
    score: float


@dataclass(frozen=True, slots=True)
class SelectionResult:
    """Resultado de la selección: pesos + decisión + rastro auditable."""

    mode: SelectionMode
    weights: dict[str, float]
    winner: str | None
    decision: str
    reason: str
    scores: tuple[ExpertNetScore, ...]