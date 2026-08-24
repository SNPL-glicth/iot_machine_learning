"""Reward de una predicción (FASE 3).

El reward es multi-dimensión (dirección x magnitud x calibración) menos
los costos de ejecución (costos + slippage + penalización por riesgo).
Solo se materializa en la transición ``EVALUATED -> REWARDED``; desde
``PENDING``, ``WAITING_OUTCOME`` o ``INVALIDATED`` nunca se produce.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .evaluation import Evaluation
    from .outcome import Outcome
    from .prediction import Prediction

_EPS = 1e-12


@dataclass(frozen=True, slots=True, kw_only=True)
class RewardConfig:
    """Pesos del reward multi-dimensión (todos >= 0).

    Attributes:
        direction_weight: Peso del acierto direccional (+1 / -1).
        magnitude_weight: Peso de la cercanía de magnitud (0..1 de acierto).
        calibration_weight: Peso de la calibración de la probabilidad.
        cost_rate: Costo de trading fijo (fracción del reward).
        slippage_rate: Slippage esperado (fracción del reward).
        risk_penalty: Penalización por riesgo asumido (fracción del reward).
    """

    direction_weight: float = 1.0
    magnitude_weight: float = 0.5
    calibration_weight: float = 0.25
    cost_rate: float = 0.0
    slippage_rate: float = 0.0
    risk_penalty: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "direction_weight",
            "magnitude_weight",
            "calibration_weight",
            "cost_rate",
            "slippage_rate",
            "risk_penalty",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} debe ser finito y >= 0: {value!r}")

    @property
    def execution_costs(self) -> float:
        """Costos totales de ejecución (costos + slippage + riesgo)."""
        return self.cost_rate + self.slippage_rate + self.risk_penalty


@dataclass(frozen=True, slots=True, kw_only=True)
class Reward:
    """Reward multi-dimensión concedido a una predicción evaluada.

    Attributes:
        direction_component: Aporte del acierto direccional.
        magnitude_component: Aporte de la cercanía de magnitud.
        calibration_component: Aporte de la calibración.
        execution_costs: Costos descontados (costos + slippage + riesgo).
        total: Suma de componentes menos costos.
    """

    direction_component: float
    magnitude_component: float
    calibration_component: float
    execution_costs: float
    total: float


def compute_reward(
    prediction: Prediction,
    outcome: Outcome,
    evaluation: Evaluation,
    config: RewardConfig,
) -> Reward:
    """Calcula el reward multi-dimensión (función pura).

    Componentes:
        dirección:  +direction_weight si direction_correct, si no -direction_weight.
        magnitud:   magnitude_weight * (1 - min(1, |err| / (|expected| + eps))).
                    Esperada 0 -> sin crédito de magnitud (0).
        calibración: calibration_weight * (1 - calibration_error).
        costos:     config.execution_costs (siempre restan).

    Casos de referencia (config por defecto):
        favorable (dirección + magnitud + calibración perfectas) -> total > 0.
        desfavorable (dirección fallada) -> total < 0 incluso sin costos.
    """
    if not isinstance(config, RewardConfig):
        raise TypeError("config debe ser RewardConfig")
    from .evaluation import Evaluation

    if not isinstance(evaluation, Evaluation):
        raise TypeError("evaluation debe ser Evaluation")

    direction_component = config.direction_weight * (
        1.0 if evaluation.direction_correct else -1.0
    )

    scale = abs(prediction.expected_return)
    if scale <= _EPS:
        magnitude_accuracy = 0.0
    else:
        magnitude_accuracy = 1.0 - min(
            1.0, evaluation.magnitude_error / scale
        )
    magnitude_component = config.magnitude_weight * magnitude_accuracy

    calibration_component = config.calibration_weight * (
        1.0 - evaluation.calibration_error
    )

    execution_costs = config.execution_costs
    total = (
        direction_component
        + magnitude_component
        + calibration_component
        - execution_costs
    )
    return Reward(
        direction_component=direction_component,
        magnitude_component=magnitude_component,
        calibration_component=calibration_component,
        execution_costs=execution_costs,
        total=total,
    )
