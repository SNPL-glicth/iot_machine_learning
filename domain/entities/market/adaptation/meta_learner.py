"""FASE 5 — MetaLearner: Hedge online por (régimen × horizonte).

Cada experto acumula ganancia = net_return REALIZADO (económico, no
accuracy); pesos ∝ softmax(η·ganancia). Frío → uniforme explícito.
Tipos en ``meta_types``, fusión en ``meta_blend``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .meta_types import GAIN_CLIP, META_STATE_VERSION, ExpertOutcome

__all__ = ["MetaLearner"]


class MetaLearner:
    """Hedge online por (régimen × horizonte), en memoria."""

    def __init__(
        self, experts: Sequence[str], *, eta: float = 200.0,
    ) -> None:
        names = [e.strip() for e in experts]
        if not names:
            raise ValueError("experts no puede ser vacío")
        if len(set(names)) != len(names):
            raise ValueError(f"experts duplicados: {names!r}")
        if not math.isfinite(eta) or eta <= 0:
            raise ValueError(f"eta debe ser > 0: {eta!r}")
        self._experts = tuple(names)
        self._eta = eta
        self._gains: dict[tuple[str, int, str], float] = {}
        self._counts: dict[tuple[str, int, str], int] = {}
        self.updates = 0

    @property
    def experts(self) -> tuple[str, ...]:
        return self._experts

    @property
    def eta(self) -> float:
        return self._eta

    def _key(self, regime: str, horizon: int, expert: str) -> tuple[str, int, str]:
        return (regime, horizon, expert)

    def update(self, outcomes: Sequence[ExpertOutcome]) -> int:
        """Incorpora outcomes realizados (retorna updates aplicados)."""
        applied = 0
        for outcome in outcomes:
            if not isinstance(outcome, ExpertOutcome):
                raise TypeError("outcomes debe contener ExpertOutcome")
            if outcome.expert not in self._experts:
                raise ValueError(
                    f"experto desconocido: {outcome.expert!r} "
                    f"(registrados: {list(self._experts)})"
                )
            gain = max(-GAIN_CLIP, min(GAIN_CLIP, outcome.net_return))
            key = self._key(outcome.regime, outcome.horizon_seconds,
                            outcome.expert)
            self._gains[key] = self._gains.get(key, 0.0) + gain
            self._counts[key] = self._counts.get(key, 0) + 1
            applied += 1
        self.updates += applied
        return applied

    def weights(self, regime: str, horizon_seconds: int) -> dict[str, float]:
        """Pesos Hedge del contexto (uniforme en frío, explícito)."""
        logits = [
            self._eta * self._gains.get(
                self._key(regime, horizon_seconds, expert), 0.0)
            for expert in self._experts
        ]
        peak = max(logits)
        exps = [math.exp(v - peak) for v in logits]
        total = sum(exps)
        return {
            expert: w / total
            for expert, w in zip(self._experts, exps, strict=True)
        }

    def counts(self, regime: str, horizon_seconds: int) -> dict[str, int]:
        """Muestras por experto en el contexto (fuerza de evidencia)."""
        return {
            expert: self._counts.get(
                self._key(regime, horizon_seconds, expert), 0)
            for expert in self._experts
        }

    def to_state(self) -> dict:
        """Estado serializable (JSON) con versión."""
        return {
            "version": META_STATE_VERSION,
            "eta": self._eta,
            "experts": list(self._experts),
            "updates": self.updates,
            "gains": {
                f"{regime}|{horizon}|{expert}": gain
                for (regime, horizon, expert), gain in self._gains.items()
            },
            "counts": {
                f"{regime}|{horizon}|{expert}": count
                for (regime, horizon, expert), count in self._counts.items()
            },
        }

    @classmethod
    def from_state(cls, state: dict) -> MetaLearner:
        """Restaura desde ``to_state`` (versión estricta)."""
        if state.get("version") != META_STATE_VERSION:
            raise ValueError(
                f"versión de estado desconocida: {state.get('version')!r}"
            )
        learner = cls(state["experts"], eta=state.get("eta", 200.0))
        for key, gain in state.get("gains", {}).items():
            regime, horizon, expert = key.split("|")
            learner._gains[(regime, int(horizon), expert)] = float(gain)
        for key, count in state.get("counts", {}).items():
            regime, horizon, expert = key.split("|")
            learner._counts[(regime, int(horizon), expert)] = int(count)
        learner.updates = int(state.get("updates", 0))
        return learner
