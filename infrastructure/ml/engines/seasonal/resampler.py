"""Linear resampling to uniform grid for irregular timestamps."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _interpolate_point(
    t: float,
    seg_ts: List[float],
    seg_values: List[float],
) -> float:
    """Interpola linealmente el valor para el timestamp t con guard clauses."""
    if t <= seg_ts[0]:
        return seg_values[0]
    if t >= seg_ts[-1]:
        return seg_values[-1]

    for j in range(len(seg_ts) - 1):
        if not (seg_ts[j] <= t <= seg_ts[j + 1]):
            continue
        dt_local = seg_ts[j + 1] - seg_ts[j]
        if dt_local == 0:
            return seg_values[j]
        frac = (t - seg_ts[j]) / dt_local
        return seg_values[j] + frac * (seg_values[j + 1] - seg_values[j])

    return seg_values[-1]


def resample_to_uniform(
    values: List[float],
    timestamps: Optional[List[float]],
) -> List[float]:
    """Resample a uniform interval via linear interpolation.

    If no timestamps → returns values unchanged.
    Target interval = median Δt. Gaps > max_gap_multiplier × median
    are treated as discontinuities; only the longest continuous segment
    is returned.
    """
    if timestamps is None or len(timestamps) != len(values):
        return values

    cfg_path = Path(__file__).parent / "config.json"
    cfg = {
        "resample_method": "linear",
        "max_gap_multiplier": 5,
        "min_points_after_resample": 6,
    }
    try:
        with cfg_path.open("r", encoding="utf-8") as fh:
            cfg = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    raw_multiplier = cfg.get("max_gap_multiplier", 5)
    max_gap_multiplier = float(raw_multiplier) if raw_multiplier is not None else 5.0
    raw_min_points = cfg.get("min_points_after_resample", 6)
    min_points = int(raw_min_points) if raw_min_points is not None else 6

    # Build continuous segments
    dts = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    if not dts:
        return values
    median_dt = float(np.median(dts))
    if median_dt <= 0:
        return values
    max_gap = max_gap_multiplier * median_dt

    segments: List[Tuple[int, int]] = []
    start = 0
    for i in range(1, len(timestamps)):
        if (timestamps[i] - timestamps[i - 1]) > max_gap:
            segments.append((start, i))
            start = i
    segments.append((start, len(timestamps)))

    longest = max(segments, key=lambda s: s[1] - s[0])
    s_start, s_end = longest
    seg_values = values[s_start:s_end]
    seg_ts = timestamps[s_start:s_end]
    if len(seg_values) < min_points:
        return seg_values

    # Linear interpolation to uniform grid
    target_dt = median_dt
    uniform_ts = [seg_ts[0] + i * target_dt for i in range(len(seg_values))]
    return [_interpolate_point(t, seg_ts, seg_values) for t in uniform_ts]
