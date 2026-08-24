"""Metrics collection and analysis for replay (FASE 5)."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class MetricKey:
    """Clave para agrupar métricas: símbolo + horizonte + régimen."""

    instrument: str
    horizon_seconds: int
    regime: str | None = None
    strategy: str | None = None

    @property
    def symbol(self) -> str:
        """Alias para compatibilidad."""
        return self.instrument


@dataclass
class MetricCollector:
    """Recolector simple de métricas por clave."""

    _data: dict[MetricKey, list[float]] = field(default_factory=dict)
    _predictions: list = field(default_factory=list)

    def add(self, key: MetricKey, pred, outcome) -> None:
        """Agrega una predicción y su outcome para calcular métricas."""
        self._predictions.append((key, pred, outcome))

    def extend(self, key: MetricKey, values: Iterable[float]) -> None:
        self._data.setdefault(key, []).extend(values)

    def get(self, key: MetricKey) -> list[float]:
        return self._data.get(key, [])

    def keys(self) -> list[MetricKey]:
        return list({p[0] for p in self._predictions})

    def clear(self) -> None:
        self._data.clear()
        self._predictions.clear()

    def totals(self):
        """Calcula métricas agregadas."""
        from ..prediction.evaluation import evaluate_prediction
        from ..prediction.reward import RewardConfig, compute_reward

        if not self._predictions:
            return _EmptyMetrics()

        n = len(self._predictions)
        total_brier = 0.0
        total_mae = 0.0
        total_rmse_sq = 0.0
        direction_correct = 0
        total_reward = 0.0

        for key, pred, outcome in self._predictions:
            evaluation = evaluate_prediction(pred, outcome)
            reward = compute_reward(pred, outcome, evaluation, RewardConfig())
            prob = pred.probability_up
            actual = 1.0 if outcome.return_realized > 0.0 else 0.0
            total_brier += (prob - actual) ** 2
            total_mae += abs(pred.expected_return - outcome.return_realized)
            total_rmse_sq += (pred.expected_return - outcome.return_realized) ** 2
            if evaluation.direction_correct:
                direction_correct += 1
            total_reward += reward.total

        return _Metrics(
            n=n,
            brier=total_brier / n,
            direction_accuracy=direction_correct / n,
            mae=total_mae / n,
            rmse=(total_rmse_sq / n) ** 0.5,
            reward=total_reward / n,
        )


@dataclass(frozen=True)
class _EmptyMetrics:
    n: int = 0
    brier: float = 0.0
    direction_accuracy: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    reward: float = 0.0


@dataclass(frozen=True)
class _Metrics:
    n: int
    brier: float
    direction_accuracy: float
    mae: float
    rmse: float
    reward: float


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    """Métricas de una predicción resuelta."""

    symbol: str
    horizon_seconds: int
    regime: str | None
    prob_up: float
    direction_correct: bool
    reward: float
    timestamp: float


def confidence_bucket(prob: float, n_buckets: int = 10) -> tuple[float, float]:
    """Bucket de confianza: devuelve (lower, upper) del bucket."""
    idx = int(prob * n_buckets)
    idx = min(idx, n_buckets - 1)
    lower = idx / n_buckets
    upper = (idx + 1) / n_buckets
    # Handle edge case for prob=0.95 -> upper=1.01 for display
    if prob >= 0.95:
        upper = 1.01
    return (lower, upper)