"""Filtro temporal de régimen (FASE 2).

Bayes de un paso: prior = Transición @ previo, posterior ∝ prior ×
verosimilitud. Da persistencia (el régimen no parpadea por una vela)
y aprende la matriz de transición con MAPs ya realizados — jamás con
futuro. Componente con estado (el único del paquete); la matemática
pura vive en ``model``/``features``.
"""

from __future__ import annotations

from .model import TransitionMatrix, uniform_posterior
from .states import LatentRegime, RegimePosterior

__all__ = ["RegimeFilter"]


class RegimeFilter:
    """Suavizado temporal + conteo de transiciones (en memoria)."""

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        smoothing: float = 1.0,
    ) -> None:
        self.posterior: RegimePosterior = uniform_posterior()
        self.transitions = TransitionMatrix(smoothing=smoothing)
        self.temperature = temperature
        self.steps_in_regime = 0
        self._updates = 0

    def update(self, likelihood: RegimePosterior) -> RegimePosterior:
        """Incorpora la verosimilitud del tramo y retorna el posterior."""
        if not isinstance(likelihood, RegimePosterior):
            raise TypeError("likelihood debe ser RegimePosterior")
        prev_map = self.posterior.most_likely
        prior = self.transitions.predict(self.posterior)
        unnormalized = [
            prior.probs[i] * likelihood.probs[i]
            for i in range(len(LatentRegime))
        ]
        total = sum(unnormalized)
        if total <= 0:
            self.posterior = uniform_posterior()
        else:
            self.posterior = RegimePosterior(
                probs=tuple(v / total for v in unnormalized)
            )
        curr_map = self.posterior.most_likely
        if self._updates > 0:
            self.transitions.observe(prev_map, curr_map)
        self.steps_in_regime = (
            self.steps_in_regime + 1 if curr_map is prev_map else 1
        )
        self._updates += 1
        return self.posterior

    @property
    def updates(self) -> int:
        """Verosimilitudes incorporadas."""
        return self._updates

    @property
    def stable_regime(self) -> LatentRegime | None:
        """MAP actual si lleva ≥3 tramos (si no, None: sin evidencia)."""
        return self.posterior.most_likely if self.steps_in_regime >= 3 else None
