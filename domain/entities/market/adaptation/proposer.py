"""WeightProposer (FASE 8) — PROPUESTA ≠ UPDATE.

El sistema primero DICE qué haría y por qué, y NO toca el modelo:

    Expert: Momentum
    Regime: TRENDING
    Horizon: 900s
    Current weight: 0.25
    Observed reward: +0.81
    Calibration: 0.72
    Sample size: 183
    Proposed weight: 0.31
    Reason: "Superior reward-adjusted performance under TRENDING/900s."

Solo si el AdaptationGuard acepta la propuesta se materializa una nueva
versión del modelo (append-only). La razón es una cadena generada con
los números reales del historial: nada de "porque la IA aprendió".

Reglas del proponedor (todas puras y documentadas):
- pesos por contexto (experto, régimen, horizonte); sin datos en un
  contexto, el peso actual se conserva (renormalizado);
- pesos propuestos ∝ softmax(reward_adjusted) entre los expertos del
  mismo contexto con muestra suficiente;
- acotado por ``max_change`` absoluto por experto;
- piso ``min_weight``: ningún experto puede desaparecer accidentalmente;
- la suma de pesos por contexto se renormaliza a 1.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from typing import Iterable

from .expert_scores import ExpertScore
from .proposal import WeightProposal
from .proposal_compute import (
    default_weights,
    softmax_target,
    bounded_update,
    reason,
)

__all__ = ["WeightProposer", "WeightProposal", "default_weights"]


class WeightProposer:
    """Genera propuestas por contexto a partir de los ExpertScores."""

    def __init__(
        self,
        *,
        min_n: int = 10,
        max_change: float = 0.10,
        min_weight: float = 0.05,
        temperature: float = 1.0,
        default_weight: float = 0.25,
    ) -> None:
        if min_n < 1:
            raise ValueError(f"min_n debe ser >= 1: {min_n}")
        if not 0.0 < max_change <= 1.0:
            raise ValueError(f"max_change inválida: {max_change}")
        if not 0.0 < min_weight <= 0.5:
            raise ValueError(f"min_weight inválida: {min_weight}")
        if temperature <= 0.0:
            raise ValueError(f"temperature debe ser > 0: {temperature}")
        self.min_n = min_n
        self.max_change = max_change
        self.min_weight = min_weight
        self.temperature = temperature
        self.default_weight = default_weight

    def propose(
        self,
        scores: tuple[ExpertScore, ...],
        current_weights: Mapping[str, dict[str, float]],
        *,
        parent_version: str | None = None,
        now: float | None = None,
    ) -> tuple[WeightProposal, ...]:
        """Propone pesos por contexto con muestra suficiente.

        Args:
            scores: ExpertScores del PerformanceAnalyzer.
            current_weights: dict ``{context_label: {expert: weight}}`` de
                la última versión del modelo (o defaults).
        """
        proposals: list[WeightProposal] = []
        contexts: dict[tuple[str | None, int], list[ExpertScore]] = {}
        for score in scores:
            contexts.setdefault((score.regime, score.horizon_seconds), []).append(score)
        for (regime, horizon), group in sorted(
            contexts.items(), key=lambda item: (item[0][0] or "", item[0][1])
        ):
            _, context_proposals = self.propose_vector(
                regime,
                horizon,
                group,
                current_weights,
                parent_version=parent_version,
                now=now,
            )
            proposals.extend(context_proposals)
        return tuple(proposals)

    def propose_vector(
        self,
        regime: str | None,
        horizon: int,
        scores: Iterable[ExpertScore],
        current_weights: Mapping[str, dict[str, float]],
        *,
        parent_version: str | None = None,
        now: float | None = None,
    ) -> tuple[dict[str, float], tuple[WeightProposal, ...]]:
        """Vector de pesos propuesto para UN contexto + sus propuestas.

        Retorna el vector completo (suma = 1, acotado, con piso) y las
        propuestas individuales derivadas: el AdaptationGuard verifica el
        vector que realmente se escribiría en la nueva versión.
        """
        group = tuple(scores)
        eligible = [s for s in group if s.n >= self.min_n]
        current = self._context_weights(current_weights, regime, horizon)
        if not eligible:
            return current, ()
        # Sin pesos registrados para el contexto: el "actual" es uniforme
        # sobre los expertos con muestra (v1 bootstrap, suma = 1).
        if current:
            baseline = {
                s.expert: current.get(s.expert, self.default_weight)
                for s in eligible
            }
        else:
            baseline = default_weights(s.expert for s in eligible)
        target = softmax_target(eligible, self.temperature)
        proposed = bounded_update(
            baseline,
            target,
            max_change=self.max_change,
            min_weight=self.min_weight,
            default_weight=self.default_weight,
        )
        proposals: list[WeightProposal] = []
        for score in sorted(eligible, key=lambda s: s.expert):
            delta = proposed[score.expert] - baseline[score.expert]
            if abs(delta) < 1e-9:
                continue
            proposals.append(
                WeightProposal(
                    expert=score.expert,
                    regime=regime,
                    horizon_seconds=horizon,
                    current_weight=baseline[score.expert],
                    proposed_weight=proposed[score.expert],
                    observed_reward=score.mean_reward,
                    calibration=score.calibration_error,
                    sample_size=score.n,
                    accuracy=score.accuracy,
                    reason=reason(score, delta),
                    created_at=now if now is not None else time.time(),
                    parent_version=parent_version,
                )
            )
        return proposed, tuple(proposals)

    # ─── helpers ───────────────────────────────────────────────────────────

    def _context_weights(
        self,
        current_weights: Mapping[str, dict[str, float]],
        regime: str | None,
        horizon: int,
    ) -> dict[str, float]:
        key = f"*|{regime or '-'}|{horizon}s"
        if key in current_weights:
            return dict(current_weights[key])
        return {}