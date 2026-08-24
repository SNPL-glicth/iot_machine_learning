"""Tests for Drift Detector get_drift_score() implementations."""

from __future__ import annotations

import pytest
import numpy as np


class TestPageHinkleyDriftScore:
    """Test PageHinkleyDetector.get_drift_score() returns bounded [0.0, 1.0]."""

    def test_get_drift_score_no_drift(self):
        from iot_machine_learning.infrastructure.ml.cognitive.drift.page_hinkley import (
            PageHinkleyDetector, PageHinkleyConfig
        )
        detector = PageHinkleyDetector(PageHinkleyConfig(delta=0.005, lambda_=50.0, alpha=0.9999))

        # Feed stable data
        for i in range(20):
            detector.update(100.0 + np.random.normal(0, 0.1))

        score = detector.get_drift_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_get_drift_score_with_drift(self):
        from iot_machine_learning.infrastructure.ml.cognitive.drift.page_hinkley import (
            PageHinkleyDetector, PageHinkleyConfig
        )
        # Use more responsive alpha (0.9) to detect drift faster
        detector = PageHinkleyDetector(PageHinkleyConfig(delta=0.005, lambda_=50.0, alpha=0.9))

        # Feed stable data first
        for i in range(20):
            detector.update(100.0)

        # Then force drift with large shift
        for i in range(100):
            detector.update(200.0 + i * 0.5)

        score = detector.get_drift_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Should be elevated after clear drift
        assert score > 0.1


class TestADWINDriftScore:
    """Test ADWINDetector.get_drift_score() returns bounded [0.0, 1.0]."""

    def test_get_drift_score_small_window(self):
        from iot_machine_learning.infrastructure.ml.cognitive.drift.adwin import ADWINDetector
        detector = ADWINDetector(delta=0.002, max_window_size=100)

        # Feed small amount of data
        for i in range(10):
            detector.update(100.0 + i * 0.1)

        score = detector.get_drift_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Small window should give small score
        assert score < 0.2

    def test_get_drift_score_full_window(self):
        from iot_machine_learning.infrastructure.ml.cognitive.drift.adwin import ADWINDetector
        detector = ADWINDetector(delta=0.002, max_window_size=50)

        # Fill the window with STABLE data (no drift)
        for i in range(50):
            detector.update(100.0 + np.random.normal(0, 0.01))

        score = detector.get_drift_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        # Full window with stable data should give score near 1.0
        assert score > 0.9


class TestErrorDriftDetectorDriftScore:
    """Test ErrorDriftDetector.get_drift_score() returns bounded [0.0, 1.0]."""

    def test_get_drift_score_normal_errors(self):
        from iot_machine_learning.infrastructure.ml.cognitive.drift.error_drift_detector import ErrorDriftDetector
        detector = ErrorDriftDetector(window_size=20, detector_type='page_hinkley')

        # Feed normal errors
        for i in range(15):
            detector.update(100.0, 100.0 + np.random.normal(0, 1))

        score = detector.get_drift_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_get_drift_score_large_errors(self):
        from iot_machine_learning.infrastructure.ml.cognitive.drift.error_drift_detector import ErrorDriftDetector
        detector = ErrorDriftDetector(window_size=20, detector_type='page_hinkley')

        # Feed large errors
        for i in range(15):
            detector.update(100.0, 150.0)  # Large consistent error

        score = detector.get_drift_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestMahalanobisFilterSingularCovariance:
    """Test MahalanobisFilter handles singular covariance matrices."""

    def test_singular_covariance_no_crash(self):
        from core.orchestration.rosa_roja.modules.module1_ingestion import MahalanobisFilter
        import numpy as np

        # Create filter with small window
        filter = MahalanobisFilter(noise_threshold=3.0, history_window=10, min_samples_for_cov=5)

        # Feed collinear data (singular covariance)
        # All points lie on a line in 3D space
        base_vector = np.array([1.0, 2.0, 3.0])
        for i in range(10):
            delta = base_vector * (1.0 + i * 0.01)  # All collinear
            movement, is_outlier = filter.process_raw_step(delta, 1.0)
            # Should not crash
            assert movement is not None

        # Should have computed covariance without LinAlgError
        assert filter._cov_inv is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])