"""Observatory helper functions."""

from __future__ import annotations

from collections.abc import Sequence

from .types import ObservationRow

EVALUATED_STATUS = "rewarded"
PENDING_STATUSES = frozenset({"pending", "active", "waiting_outcome"})


def evaluated(rows: Sequence[ObservationRow]) -> list[ObservationRow]:
    return [
        r
        for r in rows
        if r.status == EVALUATED_STATUS and r.direction_correct is not None
    ]


def chronological(rows: Sequence[ObservationRow]) -> list[ObservationRow]:
    return sorted(rows, key=lambda r: r.emitted_at)