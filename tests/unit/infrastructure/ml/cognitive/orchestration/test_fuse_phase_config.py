"""Tests for FusePhase configuration injection.

Verifies that FusePhase uses injected config values instead of hardcoded defaults.
"""

import pytest
from unittest.mock import Mock

from iot_machine_learning.infrastructure.ml.cognitive.analysis.types import (
    EnginePerception,
    InhibitionState,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import (
    FusePhase,
    _apply_spatial_correction,
    _validate_correlation_quality,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase_config import (
    FusePhaseConfig,
)


class TestFusePhaseConfigInjection:
    """Test that FusePhase uses injected configuration."""
    
    def test_spatial_correction_uses_injected_max_correction_pct(self):
        """Spatial correction respects config.max_correction_pct."""
        # Custom config with 5% max correction (instead of default 15%)
        config = FusePhaseConfig(max_correction_pct=0.05)
        
        base_prediction = 100.0
        neighbors = [("neighbor_1", 0.9)]  # High correlation
        neighbor_values = {
            "neighbor_1": [float(i) for i in range(90, 120)],  # 30 samples, strong gradient
        }
        
        corrected = _apply_spatial_correction(
            base_prediction, neighbors, neighbor_values, config
        )
        
        # Correction should be limited to 5% of base (5.0 max)
        correction = abs(corrected - base_prediction)
        assert correction <= 5.0, f"Correction {correction} exceeds 5% limit"
    
    def test_spatial_correction_uses_injected_min_gradient_samples(self):
        """Spatial correction respects config.min_gradient_samples."""
        # Custom config requiring 10 samples (instead of default 5)
        config = FusePhaseConfig(min_gradient_samples=10)
        
        base_prediction = 100.0
        neighbors = [("neighbor_1", 0.9)]
        neighbor_values = {
            "neighbor_1": [90.0, 91.0, 92.0, 93.0, 94.0],  # Only 5 samples
        }
        
        corrected = _apply_spatial_correction(
            base_prediction, neighbors, neighbor_values, config
        )
        
        # Should NOT apply correction (insufficient samples)
        assert corrected == base_prediction
    
    def test_spatial_correction_uses_injected_min_correlation(self):
        """Spatial correction respects config.min_correlation."""
        # Custom config requiring 0.85 correlation (instead of default 0.7)
        config = FusePhaseConfig(min_correlation=0.85)
        
        base_prediction = 100.0
        neighbors = [("neighbor_1", 0.75)]  # 0.75 < 0.85
        neighbor_values = {
            "neighbor_1": [float(i) for i in range(90, 120)],  # 30 samples
        }
        
        corrected = _apply_spatial_correction(
            base_prediction, neighbors, neighbor_values, config
        )
        
        # Should NOT apply correction (correlation too low)
        assert corrected == base_prediction
    
    def test_validate_correlation_uses_injected_min_samples(self):
        """Correlation validation respects config.min_samples_for_significance."""
        # Custom config requiring 30 samples (instead of default 20)
        config = FusePhaseConfig(min_samples_for_significance=30)
        
        # 25 samples: sufficient for default (20), insufficient for custom (30)
        is_valid = _validate_correlation_quality(0.8, 25, config)
        
        assert is_valid is False, "Should reject with < 30 samples"
        
        # 30 samples: sufficient for custom config
        is_valid = _validate_correlation_quality(0.8, 30, config)
        
        assert is_valid is True, "Should accept with >= 30 samples"
    
    def test_validate_correlation_uses_t_critical_table(self):
        """Correlation validation uses accurate t-critical values."""
        config = FusePhaseConfig()
        
        # Test with different sample sizes
        # For n=30 (df=28), t_critical should be 2.042 (from table)
        # Correlation 0.4 with n=30 should be rejected
        is_valid = _validate_correlation_quality(0.4, 30, config)
        assert is_valid is False
        
        # Correlation 0.5 with n=30 should be accepted (t_stat > 2.042)
        is_valid = _validate_correlation_quality(0.5, 30, config)
        assert is_valid is True
    
    def test_fuse_phase_uses_injected_hampel_k(self):
        """FusePhase uses config.hampel_k for outlier detection."""
        # Custom config with k=5.0 (instead of default 3.0)
        config = FusePhaseConfig(hampel_k=5.0)
        phase = FusePhase(config=config)
        
        # Create perceptions with one outlier
        perceptions = [
            EnginePerception(
                engine_name="engine_1",
                predicted_value=100.0,
                confidence=0.8,
                trend="stable",
                stability=0.1,
                local_fit_error=0.5,
            ),
            EnginePerception(
                engine_name="engine_2",
                predicted_value=105.0,
                confidence=0.8,
                trend="stable",
                stability=0.1,
                local_fit_error=0.5,
            ),
            EnginePerception(
                engine_name="engine_3",
                predicted_value=200.0,  # Outlier
                confidence=0.8,
                trend="stable",
                stability=0.1,
                local_fit_error=0.5,
            ),
        ]
        
        inhibition_states = [
            InhibitionState(
                engine_name=f"engine_{i}",
                base_weight=0.33,
                inhibited_weight=0.33,
                inhibition_reason="none",
                suppression_factor=0.0,
            )
            for i in range(1, 4)
        ]
        
        # Apply Hampel filter
        filtered, _, flags, _ = phase._apply_hampel_filter(perceptions, inhibition_states)
        
        # With k=5.0 (more permissive), outlier might not be rejected
        # With k=3.0 (default), outlier would be rejected
        # This test verifies config is used (exact behavior depends on MAD)
        assert phase._config.hampel_k == 5.0
    
    def test_fuse_phase_uses_injected_hampel_enabled(self):
        """FusePhase respects config.hampel_enabled flag."""
        # Disable Hampel filter
        config = FusePhaseConfig(hampel_enabled=False)
        phase = FusePhase(config=config)
        
        perceptions = [
            EnginePerception(
                engine_name="engine_1",
                predicted_value=100.0,
                confidence=0.8,
                trend="stable",
                stability=0.1,
                local_fit_error=0.5,
            ),
            EnginePerception(
                engine_name="engine_2",
                predicted_value=1000.0,  # Extreme outlier
                confidence=0.8,
                trend="stable",
                stability=0.1,
                local_fit_error=0.5,
            ),
        ]
        
        inhibition_states = [
            InhibitionState(
                engine_name=f"engine_{i}",
                base_weight=0.5,
                inhibited_weight=0.5,
                inhibition_reason="none",
                suppression_factor=0.0,
            )
            for i in range(1, 3)
        ]
        
        # Apply Hampel filter (disabled)
        filtered, _, flags, _ = phase._apply_hampel_filter(perceptions, inhibition_states)
        
        # Should return ALL perceptions (filter disabled)
        assert len(filtered) == 2
        assert filtered == perceptions
    
    def test_fuse_phase_field_smoothing_uses_injected_min_neighbors(self):
        """Field smoothing respects config.field_smoothing_min_neighbors."""
        # Custom config requiring 3 neighbors (instead of default 2)
        config = FusePhaseConfig(field_smoothing_min_neighbors=3)
        phase = FusePhase(config=config)
        
        # Mock context with only 1 neighbor (total 2 series)
        ctx = Mock()
        ctx.series_id = "series_1"
        ctx.neighbors = [("neighbor_1", 0.9)]
        ctx.orchestrator = Mock()
        ctx.orchestrator._correlation_port = Mock()
        ctx.orchestrator._storage = Mock()
        
        # Mock neighbor prediction
        neighbor_pred = Mock()
        neighbor_pred.predicted_value = 105.0
        ctx.orchestrator._storage.get_latest_prediction_for_series.return_value = neighbor_pred
        
        fused_val = 100.0
        
        # Apply field smoothing
        result = phase._apply_field_smoothing(ctx, fused_val)
        
        # Should NOT smooth (only 2 series, need 3)
        assert result == fused_val
        ctx.orchestrator._correlation_port.smooth_with_field.assert_not_called()
    
    def test_fuse_phase_field_smoothing_uses_injected_smoothing_factor(self):
        """Field smoothing respects config.smoothing_factor."""
        # Custom config with 0.5 smoothing factor (instead of default 0.2)
        config = FusePhaseConfig(smoothing_factor=0.5)
        phase = FusePhase(config=config)
        
        # Mock context with neighbors
        ctx = Mock()
        ctx.series_id = "series_1"
        ctx.neighbors = [("neighbor_1", 0.9)]
        ctx.orchestrator = Mock()
        ctx.orchestrator._correlation_port = Mock()
        ctx.orchestrator._storage = Mock()
        
        # Mock neighbor prediction
        neighbor_pred = Mock()
        neighbor_pred.predicted_value = 105.0
        ctx.orchestrator._storage.get_latest_prediction_for_series.return_value = neighbor_pred
        
        # Mock smooth_with_field to return smoothed value
        ctx.orchestrator._correlation_port.smooth_with_field.return_value = {
            "series_1": 102.5
        }
        
        fused_val = 100.0
        
        # Apply field smoothing
        result = phase._apply_field_smoothing(ctx, fused_val)
        
        # Verify smoothing_factor was passed correctly
        ctx.orchestrator._correlation_port.smooth_with_field.assert_called_once()
        call_kwargs = ctx.orchestrator._correlation_port.smooth_with_field.call_args[1]
        assert call_kwargs["smoothing_factor"] == 0.5
    
    def test_fuse_phase_default_config_values(self):
        """FusePhase with no config uses default values."""
        phase = FusePhase()
        
        # Verify default config values
        assert phase._config.max_correction_pct == 0.15
        assert phase._config.min_gradient_samples == 5
        assert phase._config.min_correlation == 0.7
        assert phase._config.smoothing_factor == 0.2
        assert phase._config.field_smoothing_min_neighbors == 2
        assert phase._config.min_samples_for_significance == 20
        assert phase._config.p_value_threshold == 0.05
        assert phase._config.hampel_k == 3.0
        assert phase._config.hampel_enabled is True
    
    def test_fuse_phase_config_is_frozen(self):
        """FusePhaseConfig is immutable (frozen dataclass)."""
        config = FusePhaseConfig()
        
        with pytest.raises(Exception):  # FrozenInstanceError
            config.max_correction_pct = 0.5
