"""Observatory recency bands and degradation detection."""

from __future__ import annotations

from collections.abc import Sequence

from ..adaptation.guard import wilson_lower_bound
from .types import ObservationRow, BandStat
from .helpers import evaluated, chronological


def recency_bands(
    rows: Sequence[ObservationRow],
    *,
    bands: int = 4,
    z: float = 1.96,
) -> tuple[BandStat, ...]:
    """Accuracy por banda de recencia (antigua → reciente).

    La degradación se lee comparando la primera banda con la última:
    si la reciente es notablemente peor, ZENIN está empeorando con el
    tiempo (la memoria lo delata).
    """
    if bands < 2:
        raise ValueError(f"bands debe ser >= 2: {bands}")
    evaluated_rows = chronological(evaluated(rows))
    total = len(evaluated_rows)
    if total == 0:
        return ()
    size = total // bands
    stats: list[BandStat] = []
    for i in range(bands):
        start = i * size
        end = total if i == bands - 1 else (i + 1) * size
        group = evaluated_rows[start:end]
        n = len(group)
        hits = sum(1 for r in group if r.direction_correct)
        rewards = [r.reward_total for r in group if r.reward_total is not None]
        stats.append(
            BandStat(
                band=i,
                n=n,
                hits=hits,
                accuracy=hits / n if n else 0.0,
                wilson_lb=wilson_lower_bound(hits, n, z=z),
                mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
            )
        )
    return tuple(stats)


def is_degraded(
    bands: Sequence[BandStat],
    *,
    tolerance: float = 0.05,
) -> bool:
    """La banda más reciente es ``tolerance`` peor que la más antigua."""
    if len(bands) < 2:
        return False
    if bands[0].n < 1 or bands[-1].n < 1:
        return False
    return bands[-1].accuracy < bands[0].accuracy - tolerance