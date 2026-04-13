"""Temporal gap detector — FASE 2.

Detects large gaps in timestamps and segments windows accordingly.

Fixes CRIT-3: Gaps temporales mal manejados.
"""

from __future__ import annotations

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GapInfo:
    """Information about detected gap."""

    gap_index: int
    gap_size: float
    median_dt: float
    gap_ratio: float


class TemporalGapDetector:
    """Detects and handles temporal gaps in time series.

    Attributes:
        _gap_threshold_multiplier: Gap threshold as multiple of median Δt
    """

    def __init__(self, gap_threshold_multiplier: float = 3.0):
        """Initialize gap detector.

        Args:
            gap_threshold_multiplier: Gap threshold (default 3× median Δt)
        """
        self._gap_threshold_multiplier = gap_threshold_multiplier

    def detect_gaps(
        self,
        timestamps: List[float],
    ) -> List[GapInfo]:
        """Detect temporal gaps in timestamps.

        Args:
            timestamps: List of timestamps

        Returns:
            List of detected gaps
        """
        if len(timestamps) < 3:
            return []

        # Compute diffs
        diffs = [
            timestamps[i] - timestamps[i - 1]
            for i in range(1, len(timestamps))
            if timestamps[i] > timestamps[i - 1]
        ]

        if not diffs:
            return []

        # Compute median Δt
        sorted_diffs = sorted(diffs)
        n = len(sorted_diffs)
        median_dt = sorted_diffs[n // 2] if n % 2 == 1 else (sorted_diffs[n // 2 - 1] + sorted_diffs[n // 2]) / 2

        if median_dt < 1e-9:
            return []

        # Detect gaps
        gap_threshold = median_dt * self._gap_threshold_multiplier
        gaps = []

        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt > gap_threshold:
                gap_ratio = dt / median_dt
                gaps.append(GapInfo(
                    gap_index=i,
                    gap_size=dt,
                    median_dt=median_dt,
                    gap_ratio=gap_ratio,
                ))

                logger.debug(
                    "temporal_gap_detected",
                    extra={
                        "gap_index": i,
                        "gap_size": round(dt, 2),
                        "median_dt": round(median_dt, 2),
                        "gap_ratio": round(gap_ratio, 2),
                    },
                )

        return gaps

    def segment_by_gaps(
        self,
        values: List[float],
        timestamps: List[float],
    ) -> List[Tuple[List[float], List[float]]]:
        """Segment time series by gaps.

        Args:
            values: List of values
            timestamps: List of timestamps

        Returns:
            List of (values_segment, timestamps_segment) tuples
        """
        if len(values) != len(timestamps):
            return [(values, timestamps)]

        gaps = self.detect_gaps(timestamps)

        if not gaps:
            return [(values, timestamps)]

        # Split at gap indices
        segments = []
        start_idx = 0

        for gap in gaps:
            gap_idx = gap.gap_index

            # Segment before gap
            if gap_idx > start_idx:
                segments.append((
                    values[start_idx:gap_idx],
                    timestamps[start_idx:gap_idx],
                ))

            start_idx = gap_idx

        # Last segment
        if start_idx < len(values):
            segments.append((
                values[start_idx:],
                timestamps[start_idx:],
            ))

        logger.info(
            "time_series_segmented",
            extra={
                "n_gaps": len(gaps),
                "n_segments": len(segments),
                "segment_sizes": [len(seg[0]) for seg in segments],
            },
        )

        return segments

    def get_largest_segment(
        self,
        values: List[float],
        timestamps: List[float],
    ) -> Tuple[List[float], List[float]]:
        """Get largest continuous segment without gaps.

        Args:
            values: List of values
            timestamps: List of timestamps

        Returns:
            (values_segment, timestamps_segment) of largest segment
        """
        segments = self.segment_by_gaps(values, timestamps)

        if not segments:
            return (values, timestamps)

        # Return largest segment
        largest = max(segments, key=lambda seg: len(seg[0]))

        logger.debug(
            "largest_segment_selected",
            extra={
                "segment_size": len(largest[0]),
                "total_points": len(values),
            },
        )

        return largest

    def compute_robust_dt(
        self,
        timestamps: List[float],
    ) -> Optional[float]:
        """Compute robust Δt excluding gaps.

        Args:
            timestamps: List of timestamps

        Returns:
            Median Δt or None if insufficient data
        """
        if len(timestamps) < 2:
            return None

        # Get largest segment (no gaps)
        _, segment_timestamps = self.get_largest_segment(
            values=list(range(len(timestamps))),  # Dummy values
            timestamps=timestamps,
        )

        if len(segment_timestamps) < 2:
            return None

        # Compute diffs in segment
        diffs = [
            segment_timestamps[i] - segment_timestamps[i - 1]
            for i in range(1, len(segment_timestamps))
            if segment_timestamps[i] > segment_timestamps[i - 1]
        ]

        if not diffs:
            return None

        # Median
        sorted_diffs = sorted(diffs)
        n = len(sorted_diffs)
        median_dt = sorted_diffs[n // 2] if n % 2 == 1 else (sorted_diffs[n // 2 - 1] + sorted_diffs[n // 2]) / 2

        return median_dt if median_dt > 1e-9 else None
