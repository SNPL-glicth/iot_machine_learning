"""Delta spike classifier — distinguishes spike, shift, and noise.

Implements ``DeltaSpikeClassificationPort`` using local statistics
and persistence checks. Parameters come from ``AnomalyDetectorConfig``.
"""

from __future__ import annotations

import math
import statistics
from typing import List, Optional

from iot_machine_learning.domain.entities.patterns.delta_spike import (
    DeltaSpikeResult,
    SpikeClassification,
)
from iot_machine_learning.domain.ports.pattern_detection_port import (
    DeltaSpikeClassificationPort,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.config import (
    AnomalyDetectorConfig,
)


class DeltaSpikeClassifier(DeltaSpikeClassificationPort):
    """Classify a detected spike as delta, shift, or noise."""

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None) -> None:
        self._config = config or AnomalyDetectorConfig()

    def classify(self, values: List[float], spike_index: int) -> DeltaSpikeResult:
        if not values or spike_index < 1 or spike_index >= len(values):
            return _noise("invalid_index")

        cfg = self._config
        pre_window = values[max(0, spike_index - cfg.delta_min_history) : spike_index]
        if len(pre_window) < 5:
            return _noise("insufficient_history")

        pre_mean = statistics.mean(pre_window)
        pre_std = statistics.stdev(pre_window) if len(pre_window) > 1 else 0.0
        delta = abs(values[spike_index] - values[spike_index - 1])

        # Noise: delta not significant
        if delta <= cfg.delta_magnitude_sigma * max(pre_std, 1e-9):
            return _noise("delta_below_sigma_threshold")

        # Evaluate persistence post-spike
        post_start = spike_index + 1
        post_end = min(len(values), post_start + cfg.delta_persistence_window)
        post_window = values[post_start:post_end]

        if not post_window:
            return _spike(delta, pre_mean, 0.0)

        post_mean = statistics.mean(post_window)
        post_deviation = abs(post_mean - pre_mean)
        persistence = min(
            1.0,
            post_deviation / max(delta, 1e-9) * (len(post_window) / cfg.delta_persistence_window),
        )

        # Shift: new level persists
        if persistence >= cfg.delta_persistence_score_threshold:
            return _shift(delta, persistence, pre_mean, post_mean)

        # Spike: transient change that reverts
        return _spike(delta, pre_mean, persistence)


def _noise(reason: str) -> DeltaSpikeResult:
    return DeltaSpikeResult(
        is_delta_spike=False,
        confidence=0.0,
        delta_magnitude=0.0,
        persistence_score=0.0,
        classification=SpikeClassification.NOISE_SPIKE,
        explanation=f"Noise — {reason}",
    )


def _spike(delta: float, baseline: float, persistence: float) -> DeltaSpikeResult:
    return DeltaSpikeResult(
        is_delta_spike=True,
        confidence=round(0.6 + (1.0 - persistence) * 0.3, 4),
        delta_magnitude=round(delta, 6),
        persistence_score=round(persistence, 4),
        classification=SpikeClassification.DELTA_SPIKE,
        explanation=f"Transient spike (delta={delta:.4f}, baseline={baseline:.4f})",
    )


def _shift(
    delta: float, persistence: float, pre_mean: float, post_mean: float
) -> DeltaSpikeResult:
    return DeltaSpikeResult(
        is_delta_spike=True,
        confidence=round(0.7 + persistence * 0.25, 4),
        delta_magnitude=round(delta, 6),
        persistence_score=round(persistence, 4),
        classification=SpikeClassification.DELTA_SPIKE,
        explanation=(
            f"Sustained level shift (delta={delta:.4f}, "
            f"pre={pre_mean:.4f}, post={post_mean:.4f})"
        ),
    )
