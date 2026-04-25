"""Tests for DeltaSpikeClassifier (Problema 2 fix).

3 cases:
1. Transient spike → DELTA_SPIKE, low persistence
2. Sustained shift → DELTA_SPIKE, high persistence
3. Noise → NOISE_SPIKE
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.entities.patterns.delta_spike import (
    SpikeClassification,
)
from iot_machine_learning.infrastructure.ml.anomaly.core.config import (
    AnomalyDetectorConfig,
)
from iot_machine_learning.infrastructure.ml.patterns.delta_spike_classifier import (
    DeltaSpikeClassifier,
)


class TestDeltaSpikeClassifier:
    def test_transient_spike(self) -> None:
        values = [20.0] * 20 + [50.0, 21.0, 20.5, 19.8] + [20.0] * 10
        cfg = AnomalyDetectorConfig(delta_persistence_window=3)
        clf = DeltaSpikeClassifier(config=cfg)
        result = clf.classify(values, spike_index=20)
        assert result.classification == SpikeClassification.DELTA_SPIKE
        assert result.is_delta_spike is True
        assert result.persistence_score < 0.6
        assert "spike" in result.explanation.lower()

    def test_sustained_shift(self) -> None:
        values = [20.0] * 20 + [50.0, 49.5, 48.8, 49.2, 50.1] + [49.0] * 10
        cfg = AnomalyDetectorConfig(delta_persistence_window=5)
        clf = DeltaSpikeClassifier(config=cfg)
        result = clf.classify(values, spike_index=20)
        assert result.classification == SpikeClassification.DELTA_SPIKE
        assert result.is_delta_spike is True
        assert result.persistence_score >= 0.6
        assert "shift" in result.explanation.lower()

    def test_noise_below_threshold(self) -> None:
        # pre-window with std ~2.0, delta=1.0 < 2*2.0 = 4.0 → noise
        pre = [20.0, 22.0, 18.0, 21.0, 19.0, 23.0, 17.0, 20.0, 22.0, 18.0,
               21.0, 19.0, 23.0, 17.0, 20.0, 22.0, 18.0, 21.0, 19.0, 20.0]
        values = pre + [21.0] + [20.5] * 10
        cfg = AnomalyDetectorConfig(delta_magnitude_sigma=2.0)
        clf = DeltaSpikeClassifier(config=cfg)
        result = clf.classify(values, spike_index=20)
        assert result.classification == SpikeClassification.NOISE_SPIKE
        assert result.is_delta_spike is False
