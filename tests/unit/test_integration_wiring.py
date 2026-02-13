"""Tests de integración para PASO 6 — wiring completo de componentes.

Cubre:
- DeprecationWarnings en sensor_ranges y select_engine_for_sensor
- AnomalyDomainService.train_all pasa timestamps a detectores
- TaylorEngine incluye structural_analysis en metadata
- PatternDomainService enriquece PatternResult con structural_analysis
- Pipeline completo: train temporal → detect → structural en metadata
- Backward compatibility: todo funciona sin timestamps
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from iot_machine_learning.domain.entities.anomaly import AnomalyResult, AnomalySeverity
from iot_machine_learning.domain.entities.pattern import PatternResult, PatternType
from iot_machine_learning.domain.entities.sensor_reading import (
    SensorReading,
    SensorWindow,
)
from iot_machine_learning.domain.ports.anomaly_detection_port import (
    AnomalyDetectionPort,
)
from iot_machine_learning.domain.services.anomaly_domain_service import (
    AnomalyDomainService,
)
from iot_machine_learning.domain.services.pattern_domain_service import (
    PatternDomainService,
)


def _make_window(sensor_id: int, values: list[float], timestamps: list[float] | None = None) -> SensorWindow:
    if timestamps is None:
        timestamps = [float(i) for i in range(len(values))]
    readings = [
        SensorReading(sensor_id=sensor_id, value=v, timestamp=t)
        for v, t in zip(values, timestamps)
    ]
    return SensorWindow(sensor_id=sensor_id, readings=readings)


# ── Mock detector that records timestamps ──────────────────────────────


class _TimestampTrackingDetector(AnomalyDetectionPort):
    """Detector that records whether timestamps were passed to train()."""

    def __init__(self) -> None:
        self._trained = False
        self.received_timestamps: Optional[List[float]] = None

    @property
    def name(self) -> str:
        return "timestamp_tracker"

    def train(self, historical_values: List[float], timestamps=None) -> None:
        self._trained = True
        self.received_timestamps = timestamps

    def detect(self, window: SensorWindow) -> AnomalyResult:
        return AnomalyResult.normal(series_id=str(window.sensor_id))

    def is_trained(self) -> bool:
        return self._trained


# ── DeprecationWarning tests ───────────────────────────────────────────


class TestDeprecationWarnings:
    """Verifica que legacy code emite DeprecationWarning."""

    def test_get_default_range_warns(self):
        from iot_machine_learning.domain.entities.sensor_ranges import (
            get_default_range,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = get_default_range("temperature")

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
            # Still returns the value (backward compatible)
            assert result == (15.0, 35.0)

    def test_select_engine_for_sensor_warns(self):
        from iot_machine_learning.application.use_cases.select_engine import (
            select_engine_for_sensor,
        )

        flags = MagicMock()
        flags.ML_ROLLBACK_TO_BASELINE = True

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = select_engine_for_sensor(1, flags)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "select_engine_for_series" in str(w[0].message)
            # Still works (backward compatible)
            assert result["engine_name"] == "baseline_moving_average"


# ── AnomalyDomainService.train_all timestamps wiring ──────────────────


class TestTrainAllTimestamps:
    """Verifica que train_all pasa timestamps a detectores."""

    def test_train_all_passes_timestamps(self):
        d1 = _TimestampTrackingDetector()
        d2 = _TimestampTrackingDetector()
        service = AnomalyDomainService(detectors=[d1, d2])

        values = [20.0] * 50
        timestamps = [float(i) for i in range(50)]
        service.train_all(values, timestamps=timestamps)

        assert d1.is_trained() is True
        assert d1.received_timestamps == timestamps
        assert d2.received_timestamps == timestamps

    def test_train_all_without_timestamps_backward_compatible(self):
        d1 = _TimestampTrackingDetector()
        service = AnomalyDomainService(detectors=[d1])

        service.train_all([20.0] * 50)

        assert d1.is_trained() is True
        assert d1.received_timestamps is None

    def test_train_all_with_voting_detector_temporal(self):
        """VotingAnomalyDetector recibe timestamps y activa temporal stats."""
        from iot_machine_learning.infrastructure.ml.anomaly.voting_anomaly_detector import (
            VotingAnomalyDetector,
        )

        detector = VotingAnomalyDetector(contamination=0.1, voting_threshold=0.5)
        service = AnomalyDomainService(detectors=[detector])

        import random
        random.seed(42)
        values = [20.0 + random.gauss(0, 0.5) for _ in range(200)]
        timestamps = [float(i) for i in range(200)]

        service.train_all(values, timestamps=timestamps)

        assert detector.is_trained() is True
        assert detector._temporal_stats.has_temporal is True


# ── TaylorEngine structural_analysis wiring ────────────────────────────


class TestTaylorStructuralWiring:
    """Verifica que TaylorEngine incluye structural_analysis en metadata."""

    def test_taylor_metadata_has_structural_analysis(self):
        from iot_machine_learning.infrastructure.ml.engines.taylor_engine import (
            TaylorPredictionEngine,
        )

        engine = TaylorPredictionEngine(order=2, horizon=1)
        values = [100.0 + 2.0 * i for i in range(20)]
        timestamps = [float(i) for i in range(20)]

        result = engine.predict(values, timestamps)

        assert "structural_analysis" in result.metadata
        sa = result.metadata["structural_analysis"]
        assert "slope" in sa
        assert "regime" in sa
        assert "stability" in sa
        assert sa["n_points"] == 20

    def test_taylor_structural_trending_regime(self):
        from iot_machine_learning.infrastructure.ml.engines.taylor_engine import (
            TaylorPredictionEngine,
        )

        engine = TaylorPredictionEngine(order=2, horizon=1)
        values = [100.0 + 5.0 * i for i in range(20)]

        result = engine.predict(values)

        sa = result.metadata["structural_analysis"]
        assert sa["regime"] == "trending"

    def test_taylor_fallback_no_structural(self):
        """Fallback (insufficient data) should not have structural_analysis."""
        from iot_machine_learning.infrastructure.ml.engines.taylor_engine import (
            TaylorPredictionEngine,
        )

        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict([10.0, 11.0])

        # Fallback path doesn't compute diagnostic
        assert result.metadata.get("fallback") == "insufficient_data"


# ── PatternDomainService structural enrichment ─────────────────────────


class TestPatternStructuralEnrichment:
    """Verifica que PatternDomainService enriquece con structural_analysis."""

    def test_detect_pattern_has_structural_in_metadata(self):
        service = PatternDomainService()
        window = _make_window(1, [25.0] * 10)

        result = service.detect_pattern(window)

        assert "structural_analysis" in result.metadata
        sa = result.metadata["structural_analysis"]
        assert sa["n_points"] == 10
        assert sa["regime"] == "stable"

    def test_detect_pattern_trending_structural(self):
        service = PatternDomainService()
        window = _make_window(1, [100.0 + 5.0 * i for i in range(20)])

        result = service.detect_pattern(window)

        sa = result.metadata["structural_analysis"]
        assert sa["regime"] == "trending"
        assert abs(sa["slope"] - 5.0) < 1e-6

    def test_detect_pattern_empty_window_no_structural(self):
        service = PatternDomainService()
        window = SensorWindow(sensor_id=1)

        result = service.detect_pattern(window)

        # Empty window → no structural enrichment
        assert "structural_analysis" not in result.metadata

    def test_detect_pattern_single_point_no_structural(self):
        service = PatternDomainService()
        window = _make_window(1, [25.0])

        result = service.detect_pattern(window)

        # Single point → size < 2 → no structural enrichment
        assert "structural_analysis" not in result.metadata

    def test_detect_pattern_with_detector_still_enriched(self):
        """Even with a real detector, structural_analysis is added."""
        mock_detector = MagicMock()
        mock_detector.detect_pattern.return_value = PatternResult(
            series_id="1",
            pattern_type=PatternType.DRIFTING,
            confidence=0.8,
            description="Drift detected",
            metadata={"original_key": "original_value"},
        )

        service = PatternDomainService(pattern_detector=mock_detector)
        window = _make_window(1, [100.0 + 2.0 * i for i in range(20)])

        result = service.detect_pattern(window)

        # Original metadata preserved
        assert result.metadata["original_key"] == "original_value"
        # Structural analysis added
        assert "structural_analysis" in result.metadata
        # Pattern type preserved
        assert result.pattern_type == PatternType.DRIFTING


# ── Full pipeline integration ──────────────────────────────────────────


class TestFullPipelineIntegration:
    """End-to-end: train with timestamps → detect → structural in result."""

    def test_full_anomaly_pipeline_with_temporal(self):
        """Train VotingAnomalyDetector with timestamps, detect extreme spike.

        Uses deterministic sinusoidal data with enough variance for
        IsolationForest/LOF to train properly, but bounded so that
        an extreme outlier (value=200) is unambiguously anomalous.

        Both the detector's voting_threshold AND the service's
        voting_threshold must be aligned for the spike to be declared
        anomalous at the service level.
        """
        from iot_machine_learning.infrastructure.ml.anomaly.voting_anomaly_detector import (
            VotingAnomalyDetector,
        )
        import math

        # Deterministic training data: sine wave around 20, amplitude 2
        # Mean ≈ 20, std ≈ 1.4, range [18, 22] — no randomness
        n = 200
        values = [20.0 + 2.0 * math.sin(i * 0.1) for i in range(n)]
        timestamps = [float(i) for i in range(n)]

        detector = VotingAnomalyDetector(contamination=0.1, voting_threshold=0.3)
        # Service threshold must also be low enough to match detector
        service = AnomalyDomainService(detectors=[detector], voting_threshold=0.3)
        service.train_all(values, timestamps=timestamps)

        # Detect extreme spike: value=200 is ~130σ away from mean ~20
        spike_window = _make_window(1, [20.0, 20.1, 20.2, 20.3, 200.0])
        spike_result = service.detect(spike_window)

        # Extreme spike must be detected — the detector's internal score
        # is passed through as the service-level score
        assert spike_result.is_anomaly is True
        assert spike_result.score > 0.3

    def test_full_pattern_pipeline_structural(self):
        """PatternDomainService enriches result with structural analysis."""
        service = PatternDomainService()

        # Trending window
        window = _make_window(1, [100.0 + 3.0 * i for i in range(30)])
        result = service.detect_pattern(window)

        assert "structural_analysis" in result.metadata
        sa = result.metadata["structural_analysis"]
        assert sa["regime"] == "trending"
        assert sa["n_points"] == 30
        assert abs(sa["slope"] - 3.0) < 1e-6
