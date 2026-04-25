"""CUSUM ramp detector (IMP-1).

Two-sided cumulative sum control chart with the classic ``k`` / ``h``
parameters. Detects a slow mean shift that would otherwise glide past a
simple 6σ threshold.

References:
    Page, E. S. (1954). "Continuous inspection schemes". Biometrika.

Pure function — no state. SRP.
"""

from __future__ import annotations

import math
from typing import List, Sequence


def detect_ramp(
    values: Sequence[float],
    *,
    k_sigma_factor: float = 0.5,
    h_sigma_factor: float = 4.0,
    min_samples: int = 4,
) -> bool:
    """Return ``True`` when either CUSUM arm exceeds ``h·σ``.

    ``k = k_sigma_factor * σ`` is the allowance / reference value. The
    two one-sided cumulative sums are::

        S⁺_t = max(0, S⁺_{t-1} + (x_t − μ) − k)
        S⁻_t = max(0, S⁻_{t-1} + (μ − x_t) − k)

    A ramp alarm fires when ``max(S⁺, S⁻) > h·σ``.

    ``σ`` is computed from the input window itself (population stdev)
    so the detector works for new series with no historical stats.
    When ``σ = 0`` or fewer than ``min_samples`` points are provided,
    returns ``False`` (no false positives on flatlines / short windows).
    """
    n = len(values)
    if n < min_samples:
        return False

    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(finite) < min_samples:
        return False

    mean = sum(finite) / len(finite)
    variance = sum((v - mean) ** 2 for v in finite) / len(finite)
    if variance <= 0.0:
        return False
    sigma = math.sqrt(variance)

    k = k_sigma_factor * sigma
    h = h_sigma_factor * sigma

    s_pos = 0.0
    s_neg = 0.0
    for v in finite:
        dev = v - mean
        s_pos = max(0.0, s_pos + dev - k)
        s_neg = max(0.0, s_neg - dev - k)
        if s_pos > h or s_neg > h:
            return True
    return False


__all__ = ["detect_ramp"]
