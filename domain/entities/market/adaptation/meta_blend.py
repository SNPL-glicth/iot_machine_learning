"""FASE 5 — MetaLearner: fusión de scores estáticos con Hedge online.

``score' = score × (1 − strength + strength × n × w_i)``: con pesos
uniformes el multiplicador es 1 (identidad, ``select_weights`` intacto);
con Hedge sesgado el ganador online se amplifica. Solo toca el score;
los campos económicos quedan intactos.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .selection_types import ExpertNetScore

__all__ = ["meta_adjusted_scores"]


def meta_adjusted_scores(
    net_scores: Sequence[ExpertNetScore],
    meta_weights: dict[str, float],
    *,
    strength: float = 0.5,
) -> tuple[ExpertNetScore, ...]:
    """Fusiona scores estáticos con Hedge online (aditivo)."""
    net_scores = tuple(net_scores)
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength fuera de [0, 1]: {strength!r}")
    n = len(net_scores)
    if n == 0:
        return ()
    names = [s.expert for s in net_scores]
    if set(names) != set(meta_weights):
        raise ValueError(
            f"meta_weights debe cubrir {names!r}, cubre "
            f"{sorted(meta_weights)}"
        )
    if not math.isclose(sum(meta_weights.values()), 1.0,
                        rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("meta_weights deben sumar 1.0")
    adjusted: list[ExpertNetScore] = []
    for score in net_scores:
        multiplier = 1.0 - strength + strength * n * meta_weights[score.expert]
        adjusted.append(ExpertNetScore(
            expert=score.expert,
            n=score.n,
            history_days=score.history_days,
            expected_return=score.expected_return,
            expected_cost=score.expected_cost,
            risk_penalty=score.risk_penalty,
            expected_net=score.expected_net,
            calibration_quality=score.calibration_quality,
            evidence_strength=score.evidence_strength,
            score=score.score * multiplier,
        ))
    return tuple(adjusted)
