"""Bootstrap for expert-level metrics (accuracy, reward, ECE)."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .sig_utils import mean, percentile

__all__ = [
    "ExpertMetricsCi",
    "bootstrap_expert_metrics",
]


@dataclass(frozen=True, slots=True)
class ExpertMetricsCi:
    """IC 95% de las métricas de un experto sobre sus filas recompensadas."""

    n: int
    accuracy: "BootstrapCi"
    mean_reward: "BootstrapCi"
    ece: "BootstrapCi"


def bootstrap_expert_metrics(
    rows: Sequence[tuple[bool, float, float]],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    max_rows: int = 3000,
    alpha: float = 0.05,
) -> ExpertMetricsCi:
    """Remuestreo de predicciones recompensadas (correct, reward, calibration).

    ``max_rows`` limita la muestra para mantener el costo acotado en
    instrumentos grandes (BTC ~10k filas por estrategia).
    """
    from .bootstrap import BootstrapCi
    data = list(rows)
    rng = random.Random(seed)
    if len(data) > max_rows:
        data = rng.sample(data, max_rows)
    n = len(data)

    def bootstrap_mean(
        pick: Callable[[tuple[bool, float, float]], float],
    ) -> BootstrapCi:
        distribution = sorted(
            mean([pick(v) for v in rng.choices(data, k=n)]) for _ in range(n_boot)
        )
        return BootstrapCi(
            point=mean([pick(v) for v in data]),
            ci_low=percentile(distribution, alpha / 2.0),
            ci_high=percentile(distribution, 1.0 - alpha / 2.0),
            n_boot=n_boot,
        )

    return ExpertMetricsCi(
        n=n,
        accuracy=bootstrap_mean(lambda row: 1.0 if row[0] else 0.0),
        mean_reward=bootstrap_mean(lambda row: row[1]),
        ece=bootstrap_mean(lambda row: row[2]),
    )