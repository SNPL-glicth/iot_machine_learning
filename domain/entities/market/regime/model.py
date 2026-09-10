"""Modelo de régimen latente (FASE 2).

Asignación blanda por prototipos documentados + matriz de transición
que aprende por conteo (Laplace). Sin fitting EM, sin dependencias:
determinista y auditable. El refit offline de prototipos es Fase 7.

Distancia² = Σ ((f − p)·scale)² con scales (1, 1, 2, 1, 2);
posterior ∝ exp(−d²/2τ²), τ = temperatura (default 1.0).
"""

from __future__ import annotations

import math
from typing import Final

from .features import RegimeFeatures
from .states import LatentRegime, RegimePosterior

__all__ = [
    "PROTOTYPES",
    "TransitionMatrix",
    "score_posterior",
    "uniform_posterior",
]

_ORDER: Final = tuple(LatentRegime)

#: Prototipos (abs_trend, vol_ln, autocorr, breakout, downside).
PROTOTYPES: Final = {
    LatentRegime.TRENDING: (1.5, 0.0, 0.35, 1.8, 0.50),
    LatentRegime.MEAN_REVERTING: (0.3, -0.2, -0.45, 1.8, 0.50),
    LatentRegime.LOW_VOL: (0.2, -1.2, 0.0, 1.8, 0.50),
    LatentRegime.HIGH_VOL: (0.8, 1.0, 0.0, 2.0, 0.55),
    LatentRegime.BREAKOUT: (1.8, 0.4, 0.1, 3.5, 0.50),
    LatentRegime.CRASH: (2.2, 0.7, 0.1, 2.5, 0.95),
}

_SCALES: Final = (1.0, 1.0, 2.0, 1.0, 2.0)


def uniform_posterior() -> RegimePosterior:
    """Prior uniforme (arranque frío del filtro)."""
    p = 1.0 / len(_ORDER)
    return RegimePosterior(probs=tuple(p for _ in _ORDER))


def score_posterior(
    features: RegimeFeatures, *, temperature: float = 1.0
) -> RegimePosterior:
    """Posterior blanda por cercanía a prototipos (verosimilitud)."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(f"temperature debe ser > 0: {temperature!r}")
    point = features.as_tuple()
    log_weights: list[float] = []
    for regime in _ORDER:
        proto = PROTOTYPES[regime]
        dist2 = sum(
            ((f - p) * s) ** 2 for f, p, s in zip(point, proto, _SCALES)
        )
        log_weights.append(-dist2 / (2.0 * temperature * temperature))
    peak = max(log_weights)
    weights = [math.exp(w - peak) for w in log_weights]
    total = sum(weights)
    return RegimePosterior(probs=tuple(w / total for w in weights))


class TransitionMatrix:
    """Matriz 6×6 que aprende transiciones por conteo (mutable, en memoria).

    Laplace α=1: ninguna transición tiene probabilidad 0 (el mercado
    siempre puede sorprender). ``observe`` registra MAP→MAP; ``predict``
    proyecta el prior a un paso (el filtro de Bayes lo usa).
    """

    def __init__(self, *, smoothing: float = 1.0) -> None:
        if not math.isfinite(smoothing) or smoothing <= 0:
            raise ValueError("smoothing debe ser > 0")
        self._smoothing = smoothing
        self._counts: dict[tuple[int, int], float] = {}
        self.transitions_observed = 0

    def observe(self, prev: LatentRegime, curr: LatentRegime) -> None:
        """Registra una transición MAP→MAP (solo pasado realizado)."""
        key = (_ORDER.index(prev), _ORDER.index(curr))
        self._counts[key] = self._counts.get(key, 0.0) + 1.0
        self.transitions_observed += 1

    def prob(self, prev: LatentRegime, curr: LatentRegime) -> float:
        """P(curr | prev) con suavizado (filas suman 1.0)."""
        i, j = _ORDER.index(prev), _ORDER.index(curr)
        n = len(_ORDER)
        row_total = self._smoothing * n + sum(
            self._counts.get((i, k), 0.0) for k in range(n)
        )
        return (self._smoothing + self._counts.get((i, j), 0.0)) / row_total

    def predict(self, posterior: RegimePosterior) -> RegimePosterior:
        """Prior a un paso: prior[j] = Σ_i P(j|i)·posterior[i]."""
        n = len(_ORDER)
        out = [0.0] * n
        for j in range(n):
            out[j] = sum(self.prob(_ORDER[i], _ORDER[j]) * posterior.probs[i]
                         for i in range(n))
        total = sum(out)
        return RegimePosterior(probs=tuple(v / total for v in out))
