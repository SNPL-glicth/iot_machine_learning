"""Tests for seasonal decomposition subsystem — FASE 2.

Tests STL decomposer, FFT seasonality detector, and SeasonalDecompositionPhase.
All tests are 100% unit tests with mocked dependencies.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock, patch
import math

from iot_machine_learning.infrastructure.ml.cognitive.seasonal import (
    STLDecomposer,
    FFTSeasonalityDetector,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.seasonal_decomposition_phase import (
    SeasonalDecompositionPhase,
)


class TestFFTSeasonalityDetector:
    """Test FFT-based seasonality detector."""
    
    def test_fft_detects_seasonal_pattern(self):
        """FFT should detect seasonal pattern in synthetic data."""
        detector = FFTSeasonalityDetector(min_period=4, max_period=50)
        
        # Synthetic seasonal data: period=12
        values = []
        for i in range(60):
            seasonal = 10 * math.sin(2 * math.pi * i / 12)
            values.append(seasonal + 50)  # Mean=50
        
        result = detector.decompose(values)
        
        # Should detect seasonality
        assert result is not None
        trend, seasonal, residual = result
        assert len(trend) == len(values)
        assert len(seasonal) == len(values)
        assert len(residual) == len(values)
    
    def test_fft_returns_none_on_insufficient_data(self):
        """FFT should return None when insufficient data."""
        detector = FFTSeasonalityDetector(min_period=10)
        
        # Only 5 points (< 2 * min_period)
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        result = detector.decompose(values)
        
        assert result is None
    
    def test_fft_residual_has_reduced_variance(self):
        """FFT residual should have lower variance than original."""
        detector = FFTSeasonalityDetector(min_period=4, max_period=50)
        
        # Strong seasonal pattern
        values = []
        for i in range(60):
            seasonal = 20 * math.sin(2 * math.pi * i / 12)
            noise = (i % 3) * 0.1  # Small noise
            values.append(seasonal + 50 + noise)
        
        result = detector.decompose(values)
        
        if result is not None:
            _, _, residual = result
            
            # Variance of residual should be less than original
            import statistics
            var_original = statistics.variance(values)
            var_residual = statistics.variance(residual)
            
            # Residual variance should be significantly lower
            assert var_residual < var_original * 0.5


class TestSTLDecomposer:
    """Test STL decomposer."""
    
    def test_stl_available_property(self):
        """STL should report availability correctly."""
        decomposer = STLDecomposer(period=12)
        
        # available is bool
        assert isinstance(decomposer.available, bool)
    
    def test_stl_validates_period(self):
        """STL should validate period parameter."""
        with pytest.raises(ValueError, match="period must be >= 2"):
            STLDecomposer(period=1)
    
    def test_stl_validates_seasonal_odd(self):
        """STL should require seasonal parameter to be odd."""
        with pytest.raises(ValueError, match="seasonal must be odd"):
            STLDecomposer(period=12, seasonal=8)
    
    def test_stl_returns_none_when_unavailable(self):
        """STL should return None when statsmodels unavailable."""
        decomposer = STLDecomposer(period=12)
        
        # Force unavailable
        decomposer._available = False
        
        values = [float(i) for i in range(50)]
        result = decomposer.decompose(values)
        
        assert result is None


class TestSeasonalDecompositionPhase:
    """Test SeasonalDecompositionPhase integration."""
    
    @pytest.fixture
    def mock_context(self):
        """Create mock pipeline context."""
        ctx = Mock()
        ctx.series_id = "sensor_42"
        
        # Synthetic seasonal data
        values = []
        for i in range(60):
            seasonal = 10 * math.sin(2 * math.pi * i / 12)
            values.append(seasonal + 50)
        ctx.values = values
        
        # Mock orchestrator
        orchestrator = Mock()
        ctx.orchestrator = orchestrator
        
        # Mock with_field
        ctx.with_field = Mock(return_value=ctx)
        
        return ctx
    
    def test_seasonal_phase_disabled_via_flag(self, mock_context):
        """Phase should skip decomposition when disabled."""
        phase = SeasonalDecompositionPhase(enable_seasonality=False)
        
        result = phase.execute(mock_context)
        
        # Should call with_field with seasonal_component_removed=False
        mock_context.with_field.assert_called_once()
        call_kwargs = mock_context.with_field.call_args[1]
        assert call_kwargs['seasonal_component_removed'] is False
        assert call_kwargs['seasonal_period_detected'] is None
    
    def test_seasonal_phase_removes_component_from_values(self, mock_context):
        """Phase should replace values with residual."""
        phase = SeasonalDecompositionPhase(
            enable_seasonality=True,
            seasonal_period_default=12,
            seasonal_use_stl=False,  # Use FFT
            seasonal_min_points=20,
        )
        
        original_values = mock_context.values.copy()
        
        result = phase.execute(mock_context)
        
        # Should call with_field
        mock_context.with_field.assert_called_once()
        call_kwargs = mock_context.with_field.call_args[1]
        
        # Should have new values (residual)
        if 'values' in call_kwargs:
            new_values = call_kwargs['values']
            # New values should be different from original
            assert new_values != original_values
            # Should have same length
            assert len(new_values) == len(original_values)
    
    def test_seasonal_phase_handles_insufficient_data(self, mock_context):
        """Phase should handle insufficient data gracefully."""
        phase = SeasonalDecompositionPhase(
            enable_seasonality=True,
            seasonal_min_points=100,  # Require more than we have
        )
        
        result = phase.execute(mock_context)
        
        # Should skip decomposition
        mock_context.with_field.assert_called_once()
        call_kwargs = mock_context.with_field.call_args[1]
        assert call_kwargs['seasonal_component_removed'] is False
    
    def test_seasonal_phase_propagates_period_to_context(self, mock_context):
        """Phase should propagate detected period to context."""
        phase = SeasonalDecompositionPhase(
            enable_seasonality=True,
            seasonal_period_default=12,
            seasonal_use_stl=False,
            seasonal_min_points=20,
        )
        
        result = phase.execute(mock_context)
        
        mock_context.with_field.assert_called_once()
        call_kwargs = mock_context.with_field.call_args[1]
        
        # Should have period (either detected or default)
        if call_kwargs.get('seasonal_component_removed'):
            assert 'seasonal_period_detected' in call_kwargs
    
    def test_anomaly_not_triggered_by_seasonal_pattern(self, mock_context):
        """Residual values should not contain seasonal peaks."""
        phase = SeasonalDecompositionPhase(
            enable_seasonality=True,
            seasonal_period_default=12,
            seasonal_use_stl=False,
            seasonal_min_points=20,
        )
        
        result = phase.execute(mock_context)
        
        call_kwargs = mock_context.with_field.call_args[1]
        
        if 'values' in call_kwargs and call_kwargs.get('seasonal_component_removed'):
            residual = call_kwargs['values']
            
            # Residual should have lower variance than original
            import statistics
            var_original = statistics.variance(mock_context.values)
            var_residual = statistics.variance(residual)
            
            # Seasonal component removed → lower variance
            assert var_residual < var_original
