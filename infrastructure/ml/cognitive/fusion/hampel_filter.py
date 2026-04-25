"""Hampel outlier filter for engine perceptions (IMP-2).

Median + k · 1.4826 · MAD (Median Absolute Deviation) rule. The
1.4826 scale factor (= 1/Φ⁻¹(0.75)) converts MAD to a σ estimate
under a Gaussian assumption, so ``k=3`` is a ≈3σ threshold.

Pure function — no state, no I/O. Only looks at ``predicted_value``;
confidence is handled upstream by ``InhibitionGate``.

Edge cases:
    * fewer than ``min_perceptions`` entries → no filtering.
    * MAD == 0 (all identical, or flat-line cluster) → no filtering.
    * ``k <= 0`` → no filtering (defensive).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from ..analysis.types import EnginePerception


MAD_GAUSSIAN_SCALE: float = 1.4826  # 1 / Φ⁻¹(0.75)


@dataclass(frozen=True)
class HampelResult:
    """Outcome of :func:`hampel_filter`.

    Attributes:
        kept: Perceptions surviving the filter (order preserved).
        rejected: ``(engine_name, predicted_value, z_score)`` triples
            for every dropped perception. ``z_score`` is
            ``|value − median| / (1.4826 · MAD)``.
        median: Median of the input ``predicted_value`` distribution.
        mad: Median absolute deviation of the same distribution.
    """

    kept: List[EnginePerception]
    rejected: List[Tuple[str, float, float]] = field(default_factory=list)
    median: float = 0.0
    mad: float = 0.0

    def to_dict(self) -> dict:
        return {
            "median": self.median,
            "mad": self.mad,
            "rejected": [
                {"engine": n, "predicted_value": v, "z_score": z}
                for (n, v, z) in self.rejected
            ],
        }


def _median(values: Sequence[float]) -> float:
    n = len(values)
    if n == 0:
        return 0.0
    ordered = sorted(values)
    mid = n // 2
    if n % 2 == 1:
        return float(ordered[mid])
    return float((ordered[mid - 1] + ordered[mid]) / 2.0)


def hampel_filter(
    perceptions: List[EnginePerception],
    *,
    k: float = 3.0,
    min_perceptions: int = 3,
) -> HampelResult:
    """Drop perceptions whose predicted_value lies beyond ``k·σ̂``.

    ``σ̂ = 1.4826 · MAD``. Order of ``kept`` matches input order.
    """
    if not perceptions:
        return HampelResult(kept=[], rejected=[], median=0.0, mad=0.0)

    if len(perceptions) < min_perceptions or k <= 0.0:
        return HampelResult(kept=list(perceptions), rejected=[], median=0.0, mad=0.0)

    predicted_values = [float(p.predicted_value) for p in perceptions]
    median = _median(predicted_values)
    deviations = [abs(v - median) for v in predicted_values]
    mad = _median(deviations)

    if mad <= 0.0:
        return HampelResult(kept=list(perceptions), rejected=[], median=median, mad=0.0)

    sigma_hat = MAD_GAUSSIAN_SCALE * mad
    threshold = k * sigma_hat

    kept: List[EnginePerception] = []
    rejected: List[Tuple[str, float, float]] = []
    for p, v in zip(perceptions, predicted_values):
        z = abs(v - median) / sigma_hat
        if abs(v - median) > threshold:
            rejected.append((p.engine_name, v, z))
        else:
            kept.append(p)

    return HampelResult(kept=kept, rejected=rejected, median=median, mad=mad)


__all__ = ["HampelResult", "hampel_filter", "MAD_GAUSSIAN_SCALE"]
