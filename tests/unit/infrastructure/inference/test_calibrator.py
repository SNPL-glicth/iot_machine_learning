"""Tests for probability calibrator."""

import pytest
import numpy as np

from iot_machine_learning.infrastructure.ml.inference.bayesian import (
    ProbabilityCalibrator,
    CalibratedScores,
)


class TestCalibratorBasic:
    """Test basic calibrator functionality."""
    
    def test_calibrator_fit(self):
        """Test basic calibrator fitting."""
        calibrator = ProbabilityCalibrator()
        
        # Overconfident scores
        raw_scores = np.array([0.2, 0.8, 0.5, 0.9])
        true_labels = np.array([0, 1, 0, 1])
        
        result = calibrator.calibrate(raw_scores, true_labels)
        
        assert isinstance(result, CalibratedScores)
        assert len(result.calibrated) == len(raw_scores)
        assert calibrator.is_fitted()
    
    def test_calibrator_transform(self):
        """Test transform after fitting."""
        calibrator = ProbabilityCalibrator()
        
        raw_scores = np.array([0.2, 0.8, 0.5, 0.9])
        true_labels = np.array([0, 1, 0, 1])
        
        calibrator.calibrate(raw_scores, true_labels)
        
        # Transform new score
        calibrated = calibrator.transform(0.7)
        
        assert 0.0 <= calibrated <= 1.0
    
    def test_unfitted_calibrator(self):
        """Test transform before fitting returns identity."""
        calibrator = ProbabilityCalibrator()
        
        raw_score = 0.7
        calibrated = calibrator.transform(raw_score)
        
        assert calibrated == raw_score
        assert not calibrator.is_fitted()


class TestCalibratorCorrections:
    """Test calibration corrections."""
    
    def test_calibration_fixes_overconfidence(self):
        """Test calibrator reduces overconfident scores."""
        calibrator = ProbabilityCalibrator()
        
        # Synthetic overconfident scores
        # Predicts 0.9 but only right 50% of the time
        raw_scores = np.array([0.9, 0.9, 0.9, 0.9])
        true_labels = np.array([1, 0, 1, 0])  # 50% correct
        
        result = calibrator.calibrate(raw_scores, true_labels)
        
        # Calibrated scores should be lower (less confident)
        assert np.mean(result.calibrated) < 0.9
    
    def test_calibration_preserves_ordering(self):
        """Test calibration preserves rank order."""
        calibrator = ProbabilityCalibrator()
        
        raw_scores = np.array([0.2, 0.5, 0.8])
        true_labels = np.array([0, 0, 1])
        
        result = calibrator.calibrate(raw_scores, true_labels)
        
        # Calibrated scores should maintain order
        assert result.calibrated[0] < result.calibrated[1] < result.calibrated[2]


class TestCalibratorEdgeCases:
    """Test edge cases."""
    
    def test_insufficient_data(self):
        """Test with insufficient training data."""
        calibrator = ProbabilityCalibrator()
        
        # Single sample
        raw_scores = np.array([0.8])
        true_labels = np.array([1])
        
        result = calibrator.calibrate(raw_scores, true_labels)
        
        # Should use identity transform
        assert calibrator.a == 1.0
        assert calibrator.b == 0.0
    
    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        calibrator = ProbabilityCalibrator()
        
        raw_scores = np.array([0.1, 0.9])
        true_labels = np.array([0, 1])
        
        result = calibrator.calibrate(raw_scores, true_labels)
        
        # Should still fit without error
        assert calibrator.is_fitted()
    
    def test_all_same_label(self):
        """Test with all same labels."""
        calibrator = ProbabilityCalibrator()
        
        raw_scores = np.array([0.2, 0.5, 0.8])
        true_labels = np.array([1, 1, 1])
        
        result = calibrator.calibrate(raw_scores, true_labels)
        
        # Should fit without error
        assert calibrator.is_fitted()


class TestCalibratorArrayHandling:
    """Test array vs scalar handling."""
    
    def test_transform_scalar(self):
        """Test transform with scalar input."""
        calibrator = ProbabilityCalibrator()
        
        raw_scores = np.array([0.2, 0.8])
        true_labels = np.array([0, 1])
        calibrator.calibrate(raw_scores, true_labels)
        
        # Transform scalar
        calibrated = calibrator.transform(0.5)
        
        assert isinstance(calibrated, float)
        assert 0.0 <= calibrated <= 1.0
    
    def test_transform_array(self):
        """Test transform with array input."""
        calibrator = ProbabilityCalibrator()
        
        raw_scores = np.array([0.2, 0.8])
        true_labels = np.array([0, 1])
        calibrator.calibrate(raw_scores, true_labels)
        
        # Transform array
        new_scores = np.array([0.3, 0.6, 0.9])
        calibrated = calibrator.transform(new_scores)
        
        assert isinstance(calibrated, np.ndarray)
        assert len(calibrated) == 3
        assert all(0.0 <= c <= 1.0 for c in calibrated)


class TestCalibratedScoresInterface:
    """Test CalibratedScores interface."""
    
    def test_get_calibrated_method(self):
        """Test get_calibrated accessor."""
        calibrator = ProbabilityCalibrator()
        
        raw_scores = np.array([0.2, 0.8])
        true_labels = np.array([0, 1])
        
        result = calibrator.calibrate(raw_scores, true_labels)
        
        # Test accessor
        cal_0 = result.get_calibrated(0)
        assert isinstance(cal_0, float)
        assert 0.0 <= cal_0 <= 1.0
    
    def test_platt_parameters(self):
        """Test Platt scaling parameters are accessible."""
        calibrator = ProbabilityCalibrator()
        
        raw_scores = np.array([0.2, 0.8, 0.5])
        true_labels = np.array([0, 1, 0])
        
        result = calibrator.calibrate(raw_scores, true_labels)
        
        # Parameters should be set
        assert result.a is not None
        assert result.b is not None
        assert isinstance(result.a, float)
        assert isinstance(result.b, float)
