"""Observatory calibration curve (reuses FASE 7.5 bucket_calibration)."""

from __future__ import annotations

from collections.abc import Sequence

from ..replay.calibration import (
    CalibrationReport,
    CalibrationThresholds,
    bucket_calibration,
)
from .types import ObservationRow
from .helpers import evaluated


def calibration_curve(
    rows: Sequence[ObservationRow],
    *,
    thresholds: CalibrationThresholds | None = None,
) -> CalibrationReport:
    """Curva declarado vs realizado por bucket de P(up) redondeado a 0.1.

    Reutiliza ``bucket_calibration`` (FASE 7.5): bucket con
    ``|declarado − realizado| > tolerance`` → FAIL (el modelo miente
    ahí); ``n < min_n`` → INSUFFICIENT (no concluye).
    """
    evaluated_rows = evaluated(rows)
    buckets: dict[float, list[ObservationRow]] = {}
    for row in evaluated_rows:
        buckets.setdefault(round(row.probability_up * 10) / 10, []).append(row)
    samples: list[tuple[str, float, int, int]] = []
    for bucket in sorted(buckets):
        group = buckets[bucket]
        n = len(group)
        declared = sum(r.probability_up for r in group) / n
        hits = sum(1 for r in group if r.direction_correct)
        samples.append((f"{bucket:.1f}", declared, hits, n))
    return bucket_calibration(samples, thresholds)