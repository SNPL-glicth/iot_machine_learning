"""Linear resampling to uniform grid for irregular timestamps."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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

    max_gap_multiplier = cfg.get("max_gap_multiplier", 5)
    min_points = cfg.get("min_points_after_resample", 6)

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
    resampled: List[float] = []
    for t in uniform_ts:
        if t <= seg_ts[0]:
            resampled.append(seg_values[0])
        elif t >= seg_ts[-1]:
            resampled.append(seg_values[-1])
        else:
            for j in range(len(seg_ts) - 1):
                if seg_ts[j] <= t <= seg_ts[j + 1]:
                    dt_local = seg_ts[j + 1] - seg_ts[j]
                    if dt_local == 0:
                        resampled.append(seg_values[j])
                    else:
                        frac = (t - seg_ts[j]) / dt_local
                        resampled.append(
                            seg_values[j]
                            + frac * (seg_values[j + 1] - seg_values[j])
                        )
                    break
    return resampled
