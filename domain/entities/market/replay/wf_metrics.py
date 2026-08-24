"""Walk-forward metrics: ModelMetrics, EdgeMetrics, weighted_model_metrics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..costs import CostModel

__all__ = [
    "ModelMetrics",
    "EdgeMetrics",
    "weighted_model_metrics",
]


@dataclass(frozen=True, slots=True)
class ModelMetrics:
    """Modelo compuesto: media ponderada por los pesos de la versión."""

    model_reward: float
    model_accuracy: float
    n: int

    @property
    def positive(self) -> bool:
        return self.model_reward > 0


def weighted_model_metrics(
    weights: Mapping[str, float],
    expert_reward: Mapping[str, float],
    expert_accuracy: Mapping[str, float],
) -> ModelMetrics:
    """reward_modelo = Σ w_e × reward_e; accuracy análoga (solo w > 0)."""
    model_reward = 0.0
    model_accuracy = 0.0
    total_weight = 0.0
    for expert, weight in weights.items():
        if weight <= 0.0:
            continue
        reward = expert_reward.get(expert)
        accuracy = expert_accuracy.get(expert)
        if reward is None or accuracy is None:
            continue
        model_reward += weight * reward
        model_accuracy += weight * accuracy
        total_weight += weight
    if total_weight <= 0.0:
        return ModelMetrics(model_reward=0.0, model_accuracy=0.0, n=0)
    return ModelMetrics(
        model_reward=model_reward / total_weight,
        model_accuracy=model_accuracy / total_weight,
        n=0,
    )


@dataclass(frozen=True, slots=True)
class EdgeMetrics:
    """Edge del modelo después de costos (FASE 9.2), por horizonte.

    ``expected_*`` usa el retorno que el modelo DECLARA (señal);
    ``realized_*`` usa el retorno que el mercado PAGÓ (outcome real).
    Ambos restan el mismo costo por predicción (modelo de costos del
    instrumento), así la comparación bruto→neto es justa.
    """

    expected_gross: float
    expected_net: float
    realized_gross: float
    realized_net: float
    cost_bps: int
    n: int