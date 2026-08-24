"""Observatory dimension stats (BY HORIZON / BY STRATEGY / BY REGIME)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from ..adaptation.guard import wilson_lower_bound
from .types import ObservationRow, DimensionStat
from .helpers import evaluated


def dimension_stats(
    rows: Sequence[ObservationRow],
    key: Callable[[ObservationRow], str],
) -> tuple[DimensionStat, ...]:
    """Accuracy + Wilson 95% + reward medio por grupo de ``key``.

    ``predictions`` cuenta todo el grupo; ``n``/``hits``/accuracy solo
    filas evaluadas (rewarded con outcome).
    """
    groups: dict[str, list[ObservationRow]] = {}
    for row in rows:
        groups.setdefault(key(row), []).append(row)
    stats: list[DimensionStat] = []
    for label in sorted(groups):
        group = groups[label]
        evaluated_rows = evaluated(group)
        n = len(evaluated_rows)
        hits = sum(1 for r in evaluated_rows if r.direction_correct)
        rewards = [r.reward_total for r in evaluated_rows if r.reward_total is not None]
        stats.append(
            DimensionStat(
                label=label,
                predictions=len(group),
                n=n,
                hits=hits,
                accuracy=hits / n if n else 0.0,
                wilson_lb=wilson_lower_bound(hits, n),
                mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
            )
        )
    return tuple(stats)