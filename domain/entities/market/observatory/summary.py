"""Observatory summary function."""

from __future__ import annotations

from collections.abc import Sequence

from ..adaptation.guard import wilson_lower_bound
from .types import ObservationRow, ObservatorySummary
from .helpers import evaluated, EVALUATED_STATUS, PENDING_STATUSES


def observatory_summary(rows: Sequence[ObservationRow]) -> ObservatorySummary:
    """Conteos por estado del ciclo de vida y métricas de las evaluadas."""
    evaluated_rows = evaluated(rows)
    n = len(evaluated_rows)
    hits = sum(1 for r in evaluated_rows if r.direction_correct)
    rewards = [r.reward_total for r in evaluated_rows if r.reward_total is not None]
    calibrations = [
        r.calibration_error for r in evaluated_rows if r.calibration_error is not None
    ]
    return ObservatorySummary(
        total=len(rows),
        evaluated=n,
        pending=sum(1 for r in rows if r.status in PENDING_STATUSES),
        invalidated=sum(1 for r in rows if r.status == "invalidated"),
        archived=sum(1 for r in rows if r.status == "archived"),
        stale=sum(1 for r in rows if r.data_status == "stale"),
        hits=hits,
        accuracy=hits / n if n else 0.0,
        wilson_lb=wilson_lower_bound(hits, n),
        mean_reward=sum(rewards) / len(rewards) if rewards else 0.0,
        mean_calibration_error=(
            sum(calibrations) / len(calibrations) if calibrations else 0.0
        ),
    )