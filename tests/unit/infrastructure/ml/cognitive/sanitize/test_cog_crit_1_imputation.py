"""Test for COG-CRIT-1: SanitizePhase imputation instead of rejection.

Reproduces the original bug where a single NaN would reject the entire window.
Confirms that the fix imputes invalid values using median from history.
"""

import math

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.sanitize.imputer import (
    Imputer,
    MedianImputer,
    MeanImputer,
)
from iot_machine_learning.infrastructure.ml.cognitive.sanitize.phase import (
    SanitizeConfig,
    SanitizePhase,
)
from iot_machine_learning.infrastructure.ml.cognitive.sanitize.series_values import (
    SeriesValuesStore,
)
from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import (
    create_initial_context,
)


class TestMedianImputer:
    """Test MedianImputer strategy (COG-CRIT-1)."""
    
    def test_impute_with_sufficient_history(self):
        """Median imputation with sufficient historical data."""
        imputer = MedianImputer(min_history=3)
        history = [10.0, 12.0, 14.0, 11.0, 13.0]
        result = imputer.impute(float('nan'), history)
        assert result == 12.0  # Median of [10, 11, 12, 13, 14]
    
    def test_impute_with_inf(self):
        """Median imputation with Inf value."""
        imputer = MedianImputer(min_history=3)
        history = [10.0, 12.0, 14.0]
        result = imputer.impute(float('inf'), history)
        assert result == 12.0
    
    def test_impute_with_insufficient_history(self):
        """Raises error when history insufficient."""
        imputer = MedianImputer(min_history=3)
        history = [10.0, 12.0]
        with pytest.raises(ValueError, match="Insufficient history"):
            imputer.impute(float('nan'), history)
    
    def test_impute_with_fallback(self):
        """Uses fallback value when history insufficient."""
        imputer = MedianImputer(min_history=3, fallback_value=0.0)
        history = [10.0]
        result = imputer.impute(float('nan'), history)
        assert result == 0.0
    
    def test_impute_filters_invalid_history(self):
        """Filters NaN/Inf from history before computing median."""
        imputer = MedianImputer(min_history=3)
        history = [10.0, float('nan'), 12.0, float('inf'), 14.0]
        result = imputer.impute(float('nan'), history)
        assert result == 12.0  # Median of [10, 12, 14]


class TestMeanImputer:
    """Test MeanImputer strategy (OCP: interchangeable strategy)."""
    
    def test_impute_with_sufficient_history(self):
        """Mean imputation with sufficient historical data."""
        imputer = MeanImputer(min_history=3)
        history = [10.0, 12.0, 14.0]
        result = imputer.impute(float('nan'), history)
        assert result == 12.0  # Mean of [10, 12, 14]


class TestSanitizePhaseImputation:
    """Test SanitizePhase with imputation (COG-CRIT-1)."""
    
    def test_original_bug_single_nan_rejects_window(self):
        """
        COG-CRIT-1: Reproduces original bug.
        
        Before fix: A single NaN would reject the entire window.
        After fix: NaN is imputed, other valid values are preserved.
        """
        # Create mock store with historical data
        store = SeriesValuesStore(max_series=100, window_size=20)
        series_id = "test_sensor"
        history = [10.0, 12.0, 14.0, 11.0, 13.0]
        for v in history:
            store.append(series_id, v)
        
        # Create phase with imputer
        phase = SanitizePhase(
            config=SanitizeConfig(),
            series_values_store=store,
        )
        
        # Create context with one NaN in the middle
        values = [10.0, 12.0, float('nan'), 14.0, 11.0]
        ctx = create_initial_context(
            series_id=series_id,
            values=values,
            timestamps=[i * 60.0 for i in range(len(values))],
        )
        
        # Execute phase
        result = phase.execute(ctx)
        
        # BUG FIX: Should NOT reject entire window
        assert not result.is_fallback
        assert result.fallback_reason is None
        
        # Should have imputed the NaN
        assert "value_imputed" in " ".join(result.sanitization_flags)
        
        # Should preserve other valid values
        assert len(result.values) == 5  # All values preserved
        assert result.values[0] == 10.0
        assert result.values[1] == 12.0
        assert math.isfinite(result.values[2])  # NaN was imputed
        assert result.values[3] == 14.0
        assert result.values[4] == 11.0
        
        # Confidence multiplier should be reduced due to imputation
        assert result.confidence_multiplier < 1.0
        assert result.confidence_multiplier >= 0.5  # Minimum 0.5
    
    def test_multiple_nans_all_imputed(self):
        """Multiple NaN values are all imputed with appropriate penalty."""
        store = SeriesValuesStore(max_series=100, window_size=20)
        series_id = "test_sensor"
        history = [10.0, 12.0, 14.0, 11.0, 13.0]
        for v in history:
            store.append(series_id, v)
        
        phase = SanitizePhase(
            config=SanitizeConfig(),
            series_values_store=store,
        )
        
        # Multiple NaN values
        values = [10.0, float('nan'), 12.0, float('nan'), 14.0]
        ctx = create_initial_context(
            series_id=series_id,
            values=values,
            timestamps=[i * 60.0 for i in range(len(values))],
        )
        
        result = phase.execute(ctx)
        
        # Should not reject
        assert not result.is_fallback
        
        # Should have imputed 2 values
        imputed_flags = [f for f in result.sanitization_flags if "value_imputed" in f]
        assert len(imputed_flags) == 2
        
        # Confidence multiplier should be lower (2 * 0.1 = 0.2 penalty)
        assert result.confidence_multiplier == 0.8  # 1.0 - 0.2
    
    def test_no_history_rejects_invalid_values_only(self):
        """When no history available, invalid values are rejected but valid ones preserved."""
        store = SeriesValuesStore(max_series=100, window_size=20)
        series_id = "test_sensor"
        
        phase = SanitizePhase(
            config=SanitizeConfig(),
            series_values_store=store,
        )
        
        values = [10.0, 12.0, float('nan'), 14.0, 11.0]
        ctx = create_initial_context(
            series_id=series_id,
            values=values,
            timestamps=[i * 60.0 for i in range(len(values))],
        )
        
        result = phase.execute(ctx)
        
        # Should not reject entire window
        assert not result.is_fallback
        
        # Should have rejected only the NaN
        assert "value_rejected:no_history" in " ".join(result.sanitization_flags)
        
        # Should preserve valid values
        assert len(result.values) == 4  # One value rejected
        assert result.values == [10.0, 12.0, 14.0, 11.0]
    
    def test_all_values_rejected_fallback(self):
        """When all values are invalid and cannot be imputed, fallback."""
        store = SeriesValuesStore(max_series=100, window_size=20)
        series_id = "test_sensor"
        
        phase = SanitizePhase(
            config=SanitizeConfig(),
            series_values_store=store,
        )
        
        # All NaN values
        values = [float('nan'), float('nan'), float('nan')]
        ctx = create_initial_context(
            series_id=series_id,
            values=values,
            timestamps=[i * 60.0 for i in range(len(values))],
        )
        
        result = phase.execute(ctx)
        
        # Should fallback
        assert result.is_fallback
        assert result.fallback_reason == "all_values_rejected"
        assert "all_values_rejected" in result.sanitization_flags
    
    def test_no_invalid_values_no_imputation(self):
        """When all values are valid, no imputation occurs."""
        store = SeriesValuesStore(max_series=100, window_size=20)
        series_id = "test_sensor"
        history = [10.0, 12.0, 14.0]
        for v in history:
            store.append(series_id, v)
        
        phase = SanitizePhase(
            config=SanitizeConfig(),
            series_values_store=store,
        )
        
        values = [10.0, 12.0, 14.0, 11.0, 13.0]
        ctx = create_initial_context(
            series_id=series_id,
            values=values,
            timestamps=[i * 60.0 for i in range(len(values))],
        )
        
        result = phase.execute(ctx)
        
        # No imputation
        imputed_flags = [f for f in result.sanitization_flags if "imputed" in f]
        assert len(imputed_flags) == 0
        
        # Confidence multiplier should be 1.0 (no penalty)
        assert result.confidence_multiplier == 1.0


class TestImputerOCP:
    """Test OCP principle: Imputer is extensible without modification."""
    
    def test_custom_imputer_strategy(self):
        """Custom imputer can be used without modifying SanitizePhase."""
        class CustomImputer(Imputer):
            """Custom imputer that always returns 42.0."""
            def impute(self, value: float, history: List[float]) -> float:
                return 42.0
        
        store = SeriesValuesStore(max_series=100, window_size=20)
        series_id = "test_sensor"
        history = [10.0, 12.0, 14.0]
        for v in history:
            store.append(series_id, v)
        
        phase = SanitizePhase(
            config=SanitizeConfig(),
            series_values_store=store,
            imputer=CustomImputer(),  # Custom strategy
        )
        
        values = [10.0, float('nan'), 14.0]
        ctx = create_initial_context(
            series_id=series_id,
            values=values,
            timestamps=[i * 60.0 for i in range(len(values))],
        )
        
        result = phase.execute(ctx)
        
        # Custom imputer should have been used
        assert result.values[1] == 42.0
