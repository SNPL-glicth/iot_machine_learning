"""Tests for drift detection subsystem — FASE 1.

Tests Page-Hinkley detector, ADWIN detector, and DriftDetectionPhase.
All tests are 100% unit tests with mocked dependencies.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass

from iot_machine_learning.infrastructure.ml.cognitive.drift import (
    PageHinkleyDetector,
    PageHinkleyConfig,
    ADWINDetector,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.drift_detection_phase import (
    DriftDetectionPhase,
)


class TestPageHinkleyDetector:
    """Test Page-Hinkley drift detector."""
    
    def test_no_drift_on_stationary_signal(self):
        """Page-Hinkley should not detect drift on stationary signal."""
        config = PageHinkleyConfig(delta=0.005, lambda_=50.0, alpha=0.0)
        detector = PageHinkleyDetector(config)
        
        # Stationary signal around mean=10
        for _ in range(100):
            drift = detector.update(10.0)
            assert not drift
        
        assert detector.cumsum < config.lambda_
    
    def test_drift_detected_on_abrupt_shift(self):
        """Page-Hinkley should detect drift on abrupt mean shift."""
        config = PageHinkleyConfig(delta=0.005, lambda_=10.0, alpha=0.0)
        detector = PageHinkleyDetector(config)
        
        # Stationary at mean=10
        for _ in range(50):
            detector.update(10.0)
        
        # Abrupt shift to mean=15
        drift_detected = False
        for _ in range(50):
            if detector.update(15.0):
                drift_detected = True
                break
        
        assert drift_detected
    
    def test_reset_clears_state(self):
        """Reset should clear detector state."""
        config = PageHinkleyConfig(delta=0.005, lambda_=50.0, alpha=0.0)
        detector = PageHinkleyDetector(config)
        
        for _ in range(10):
            detector.update(10.0)
        
        assert detector.n_observations > 0
        assert detector.mean > 0
        
        detector.reset()
        
        assert detector.n_observations == 0
        assert detector.cumsum == 0.0
        assert detector.mean == 0.0
    
    def test_single_observation_no_drift(self):
        """Single observation should not trigger drift."""
        config = PageHinkleyConfig(delta=0.005, lambda_=50.0, alpha=0.0)
        detector = PageHinkleyDetector(config)
        
        drift = detector.update(10.0)
        
        assert not drift
        assert detector.n_observations == 1


class TestADWINDetector:
    """Test ADWIN drift detector."""
    
    def test_no_drift_on_stationary_signal(self):
        """ADWIN should not detect drift on stationary signal."""
        detector = ADWINDetector(delta=0.002, max_window_size=100)
        
        for _ in range(50):
            drift = detector.update(10.0)
            assert not drift
    
    def test_drift_detected_on_distribution_change(self):
        """ADWIN should detect drift on distribution change."""
        detector = ADWINDetector(delta=0.002, max_window_size=100)
        
        # Stationary at mean=10
        for _ in range(30):
            detector.update(10.0)
        
        # Shift to mean=20
        drift_detected = False
        for _ in range(30):
            if detector.update(20.0):
                drift_detected = True
                break
        
        assert drift_detected
    
    def test_window_shrinks_on_drift(self):
        """ADWIN should shrink window when drift detected."""
        detector = ADWINDetector(delta=0.002, max_window_size=100)
        
        for _ in range(50):
            detector.update(10.0)
        
        initial_size = detector.window_size
        
        # Trigger drift
        for _ in range(20):
            detector.update(50.0)
        
        # Window should have shrunk at some point
        assert detector.window_size < initial_size or detector.window_size < 50


class TestDriftDetectionPhase:
    """Test DriftDetectionPhase integration."""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock pipeline context."""
        ctx = Mock()
        ctx.series_id = "sensor_42"
        ctx.regime = "STABLE"
        
        # Mock signal profile
        profile = Mock()
        profile.noise_ratio = 0.1
        profile.stability = 0.9
        ctx.profile = profile
        
        # Mock orchestrator with plasticity tracker
        orchestrator = Mock()
        plasticity = Mock()
        plasticity.has_history = Mock(return_value=True)
        plasticity.reset_regime = Mock()
        orchestrator._plasticity = plasticity
        orchestrator._audit = None
        ctx.orchestrator = orchestrator
        
        # Mock with_field
        ctx.with_field = Mock(return_value=ctx)
        
        return ctx
    
    def test_drift_not_detected_on_stable_signal(self, mock_context):
        """Phase should not detect drift on stable signal."""
        phase = DriftDetectionPhase(
            enable_drift_detection=True,
            drift_lambda=50.0,
        )
        
        result = phase.execute(mock_context)
        
        # Should call with_field with drift_detected=False
        mock_context.with_field.assert_called_once()
        call_kwargs = mock_context.with_field.call_args[1]
        assert call_kwargs['drift_detected'] is False
    
    def test_drift_phase_resets_weights_on_confirmed_drift(self, mock_context):
        """Phase should reset BayesianWeightTracker when drift confirmed."""
        # Configure for easy drift detection
        phase = DriftDetectionPhase(
            enable_drift_detection=True,
            drift_delta=0.001,
            drift_lambda=0.5,  # Low threshold
            drift_alpha=0.0,  # No forgetting
            cooldown_seconds=0.0,  # No cooldown
        )
        
        # Warm up detector with stable values
        mock_context.profile.noise_ratio = 0.1
        mock_context.profile.stability = 0.9
        for _ in range(10):
            phase.execute(mock_context)
        
        # Now inject high drift score to trigger detection
        mock_context.profile.noise_ratio = 10.0
        mock_context.profile.stability = 0.01
        
        # Reset mock to count only this call
        mock_context.orchestrator._plasticity.reset_regime.reset_mock()
        
        # Execute multiple times to accumulate drift
        for _ in range(20):
            phase.execute(mock_context)
        
        # Should have called reset_regime at least once
        assert mock_context.orchestrator._plasticity.reset_regime.call_count >= 1
        # Verify it was called with correct args
        mock_context.orchestrator._plasticity.reset_regime.assert_called_with(
            regime="STABLE",
            series_id="sensor_42",
        )
    
    def test_drift_phase_respects_cooldown(self, mock_context):
        """Phase should respect cooldown between resets."""
        phase = DriftDetectionPhase(
            enable_drift_detection=True,
            drift_delta=0.001,
            drift_lambda=0.5,
            drift_alpha=0.0,
            cooldown_seconds=300.0,
        )
        
        # Warm up with stable
        mock_context.profile.noise_ratio = 0.1
        mock_context.profile.stability = 0.9
        for _ in range(10):
            phase.execute(mock_context)
        
        # High drift score
        mock_context.profile.noise_ratio = 10.0
        mock_context.profile.stability = 0.01
        
        # Execute until drift detected
        mock_context.orchestrator._plasticity.reset_regime.reset_mock()
        for _ in range(30):
            phase.execute(mock_context)
            if mock_context.orchestrator._plasticity.reset_regime.call_count > 0:
                break
        
        first_call_count = mock_context.orchestrator._plasticity.reset_regime.call_count
        assert first_call_count >= 1, "Drift should have been detected"
        
        # Second call within cooldown should NOT reset again
        mock_context.orchestrator._plasticity.reset_regime.reset_mock()
        for _ in range(10):
            phase.execute(mock_context)
        assert mock_context.orchestrator._plasticity.reset_regime.call_count == 0
    
    def test_drift_phase_handles_missing_profile(self, mock_context):
        """Phase should handle missing profile gracefully."""
        phase = DriftDetectionPhase(enable_drift_detection=True)
        
        mock_context.profile = None
        
        result = phase.execute(mock_context)
        
        # Should skip detection
        mock_context.with_field.assert_called_once()
        call_kwargs = mock_context.with_field.call_args[1]
        assert call_kwargs['drift_detected'] is False
    
    def test_drift_phase_disabled_via_flag(self, mock_context):
        """Phase should skip detection when disabled."""
        phase = DriftDetectionPhase(enable_drift_detection=False)
        
        result = phase.execute(mock_context)
        
        # Should immediately return with drift_detected=False
        mock_context.with_field.assert_called_once()
        call_kwargs = mock_context.with_field.call_args[1]
        assert call_kwargs['drift_detected'] is False
        assert call_kwargs['drift_magnitude'] == 0.0
    
    def test_drift_phase_handles_plasticity_error(self, mock_context):
        """Phase should handle plasticity reset errors gracefully."""
        phase = DriftDetectionPhase(
            enable_drift_detection=True,
            drift_lambda=0.5,
            drift_alpha=0.0,
            cooldown_seconds=0.0,
        )
        
        # Warm up
        mock_context.profile.noise_ratio = 0.1
        mock_context.profile.stability = 0.9
        for _ in range(10):
            phase.execute(mock_context)
        
        # Configure for drift detection
        mock_context.profile.noise_ratio = 10.0
        mock_context.profile.stability = 0.01
        
        # Make reset_regime raise exception
        mock_context.orchestrator._plasticity.reset_regime.side_effect = Exception("DB error")
        
        # Should not propagate exception
        for _ in range(30):
            try:
                result = phase.execute(mock_context)
            except Exception:
                pytest.fail("Phase should not propagate exceptions")
        
        # Should still have called with_field
        assert mock_context.with_field.called
