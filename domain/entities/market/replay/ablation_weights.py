"""Ablation weight computation (FASE 9.3)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

from .ablation_constants import ABLATIONS, _BASELINE_EXPERTS


class ExpertScoreLike(Protocol):
    """Mínimo contrato de un ExpertScore para resolver pesos (FASE 8)."""

    expert: str
    regime: str | None
    reward_adjusted: float
    expected_return: float


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


def _scoped(
    scores: Sequence[ExpertScoreLike], regime: str | None
) -> list[ExpertScoreLike]:
    if regime is None:
        return list(scores)
    return [s for s in scores if s.regime == regime]


def _uniform(scoped: Sequence[ExpertScoreLike]) -> dict[str, float]:
    return {s.expert: 1.0 / len(scoped) for s in scoped}


def ablation_weights(
    ablation: str,
    *,
    weights_by_context: Mapping[str, Mapping[str, float]],
    regime: str | None,
    horizon: int,
    scores: Sequence[ExpertScoreLike],
) -> dict[str, float] | None:
    """Pesos de la ablación para (régimen, horizonte) — None si sin muestra.

    Baselines: el experto solo. ZENIN - memoria: uniforme (sin historial
    de outcomes → sin adaptación). ZENIN - régimen: contexto global sobre
    todos los expertos. ZENIN - MoE: contexto resuelto + hard max sobre
    ``reward_adjusted`` (empate: orden alfabético). ZENIN completo:
    contexto resuelto con la versión activa (igual que ``evaluate_window``).
    """
    if ablation in _BASELINE_EXPERTS:
        expert = _BASELINE_EXPERTS[ablation]
        if not any(s.expert == expert for s in _scoped(scores, regime)):
            return None
        return {expert: 1.0}

    scoped = _scoped(scores, regime)
    if not scoped:
        return None

    if ablation == "ZENIN - memoria":
        return _uniform(scoped)

    if ablation == "ZENIN - régimen":
        global_weights = _context_weights(weights_by_context, None, horizon)
        if global_weights:
            names = {s.expert for s in scoped}
            return {e: w for e, w in global_weights.items() if e in names and w > 0.0}
        return _uniform(scoped)

    weights = _context_weights(weights_by_context, regime, horizon)
    if not weights:
        weights = _uniform(scoped)

    if ablation == "ZENIN - MoE":
        best = max(scoped, key=lambda s: (s.reward_adjusted, s.expert))
        return {best.expert: 1.0}

    names = {s.expert for s in scoped}
    return {e: w for e, w in weights.items() if e in names and w > 0.0}