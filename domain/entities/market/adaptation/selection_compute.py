"""FASE 9.4 — Selection Computation (net scores, softmax)."""

from __future__ import annotations

import math
from collections.abc import Sequence

from iot_machine_learning.domain.entities.market.adaptation.expert_scores import ExpertScore
from iot_machine_learning.domain.entities.market.costs import CostModel
from .selection_types import ExpertNetScore


def expert_net_scores(
    scores: Sequence[ExpertScore],
    *,
    cost_model: CostModel,
    risk_aversion: float = 0.1,
    min_n: int = 10,
) -> tuple[ExpertNetScore, ...]:
    """Convierte ExpertScores en scores netos (expected net return).

    ``expected_net = expected_return − costo − penalidad de riesgo``;
    ``score = expected_net × calibration_quality × evidence_strength``.
    La accuracy NO entra al score (métrica secundaria/guardrail).
    """
    cost = cost_model.total()
    net_scores: list[ExpertNetScore] = []
    for score in scores:
        risk_penalty = risk_aversion * score.risk_std
        expected_net = score.expected_return - cost - risk_penalty
        calibration_quality = 1.0 - min(score.calibration_error, 1.0)
        evidence_strength = min(score.n / min_n, 1.0)
        net_scores.append(
            ExpertNetScore(
                expert=score.expert,
                n=score.n,
                history_days=score.history_days,
                expected_return=score.expected_return,
                expected_cost=cost,
                risk_penalty=risk_penalty,
                expected_net=expected_net,
                calibration_quality=calibration_quality,
                evidence_strength=evidence_strength,
                score=expected_net * calibration_quality * evidence_strength,
            )
        )
    return tuple(net_scores)


def softmax(
    net_scores: Sequence[ExpertNetScore], temperature: float
) -> dict[str, float]:
    values = [s.score / temperature for s in net_scores]
    max_value = max(values)
    exp_values = [math.exp(v - max_value) for v in values]
    total = sum(exp_values)
    return {
        s.expert: exp / total
        for s, exp in zip(net_scores, exp_values, strict=True)
    }


def has_evidence(best: ExpertNetScore, config: "SelectionConfig") -> bool:
    return best.n >= config.min_n and best.history_days >= config.min_history_days