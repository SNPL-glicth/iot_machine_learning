"""Walk-forward evaluation: HorizonEval, WfRow, evaluate_window."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from ..costs import CostModel
from .wf_metrics import ModelMetrics, EdgeMetrics, weighted_model_metrics

__all__ = ["HorizonEval", "WfRow", "evaluate_window"]


class Scored(Protocol):
    """Mínimo contrato de un ExpertScore para evaluar el TEST."""

    expert: str
    regime: str | None
    horizon_seconds: int
    n: int
    accuracy: float
    mean_reward: float
    reward_adjusted: float
    expected_return: float = 0.0
    realized_return: float = 0.0


@dataclass(frozen=True, slots=True)
class HorizonEval:
    """Evaluación del TEST para un (régimen, horizonte) del modelo."""

    horizon_seconds: int
    regime: str | None
    n: int
    weights: dict[str, float]
    experts: dict[str, dict[str, float]]
    model: ModelMetrics
    edge: EdgeMetrics | None = None


@dataclass(frozen=True, slots=True)
class WfRow:
    """Fila del reporte: una ventana de un instrumento."""

    index: int
    symbol: str
    regime: str | None
    train_start: float
    train_end: float
    test_start: float
    test_end: float
    n_train: int
    horizons: tuple[HorizonEval, ...]
    accepted: int
    rejected: int
    note: str = ""
    cost_bps: int = 0
    sharpe: float | None = None
    edge_class: str | None = None

    @property
    def n_test(self) -> int:
        return sum(h.n for h in self.horizons)

    @property
    def model_reward(self) -> float:
        n = sum(h.n for h in self.horizons) or 1
        return sum(h.model.model_reward * h.n for h in self.horizons) / n

    @property
    def model_accuracy(self) -> float:
        n = sum(h.n for h in self.horizons) or 1
        return sum(h.model.model_accuracy * h.n for h in self.horizons) / n

    @property
    def realized_gross(self) -> float | None:
        """Edge bruto realizado ponderado por n (None si sin edge)."""
        edges = [h.edge for h in self.horizons if h.edge is not None]
        if not edges:
            return None
        n = sum(e.n for e in edges) or 1
        return sum(e.realized_gross * e.n for e in edges) / n

    @property
    def realized_net(self) -> float | None:
        edges = [h.edge for h in self.horizons if h.edge is not None]
        if not edges:
            return None
        n = sum(e.n for e in edges) or 1
        return sum(e.realized_net * e.n for e in edges) / n


def evaluate_window(
    scores: Sequence[Scored],
    weights_by_context: Mapping[str, Mapping[str, float]],
    *,
    regime: str | None,
    cost_model: CostModel | None = None,
) -> tuple[HorizonEval, ...]:
    """Convierte ExpertScores del TEST en HorizonEvals con el modelo.

    ``weights_by_context``: contextos de la versión activa (claves
    ``*|régimen|hhss``); se busca primero el contexto exacto del régimen
    de la ventana y se cae al contexto global (``*|-|hhss``).

    Con ``cost_model`` (FASE 9.2) cada HorizonEval lleva su ``edge``:
    retorno esperado/realizado ponderado por los pesos del modelo, bruto
    y neto de costos. La clasificación (edge_class) se hace a nivel
    ventana en el runner (necesita el sharpe por predicción).
    """
    by_horizon: dict[int, list[Scored]] = {}
    for score in scores:
        by_horizon.setdefault(score.horizon_seconds, []).append(score)

    evals: list[HorizonEval] = []
    for horizon, group in sorted(by_horizon.items()):
        if regime is not None:
            scoped = [s for s in group if s.regime == regime]
        else:
            scoped = [s for s in group if s.regime is None]
        if not scoped:
            continue
        weights = _context_weights(weights_by_context, regime, horizon)
        if not weights:
            continue
        expert_reward = {s.expert: s.mean_reward for s in scoped}
        expert_accuracy = {s.expert: s.accuracy for s in scoped}
        experts = {
            s.expert: {
                "n": s.n,
                "accuracy": s.accuracy,
                "mean_reward": s.mean_reward,
                "reward_adjusted": s.reward_adjusted,
            }
            for s in scoped
        }
        n = sum(s.n for s in scoped)
        evals.append(
            HorizonEval(
                horizon_seconds=horizon,
                regime=regime,
                n=n,
                weights=dict(weights),
                experts=experts,
                model=weighted_model_metrics(weights, expert_reward, expert_accuracy),
                edge=_edge_metrics(weights, scoped, cost_model),
            )
        )
    return tuple(evals)


def _edge_metrics(
    weights: Mapping[str, float],
    scoped: Sequence[Scored],
    cost_model: CostModel | None,
) -> EdgeMetrics | None:
    """Edge ponderado por los pesos del modelo (FASE 9.2)."""
    if cost_model is None:
        return None
    expected_gross = 0.0
    realized_gross = 0.0
    total_weight = 0.0
    n = 0
    for expert, weight in weights.items():
        if weight <= 0.0:
            continue
        score = next((s for s in scoped if s.expert == expert), None)
        if score is None:
            continue
        expected_gross += weight * score.expected_return
        realized_gross += weight * score.realized_return
        total_weight += weight
        n += score.n
    if total_weight <= 0.0:
        return None
    expected_gross /= total_weight
    realized_gross /= total_weight
    cost = cost_model.total()
    return EdgeMetrics(
        expected_gross=expected_gross,
        expected_net=expected_gross - cost,
        realized_gross=realized_gross,
        realized_net=realized_gross - cost,
        cost_bps=int(cost_model.total_bps),
        n=n,
    )


def _context_weights(
    weights_by_context: Mapping[str, Mapping[str, float]],
    regime: str | None,
    horizon: int,
) -> dict[str, float]:
    exact = f"*|{regime or '-'}|{horizon}s"
    if exact in weights_by_context:
        return dict(weights_by_context[exact])
    fallback = f"*|-|{horizon}s"
    if fallback in weights_by_context:
        return dict(weights_by_context[fallback])
    return {}