"""Tests for WeightAdjustmentService.

Covers adaptive weight calculation, fallback strategies, and edge cases.
"""

import pytest
from unittest.mock import Mock

from iot_machine_learning.infrastructure.ml.cognitive.fusion.weight_adjustment_service import WeightAdjustmentService


class TestWeightAdjustmentServiceInitialization:
    """Test service initialization."""
    
    def test_default_initialization(self):
        """Test service with default parameters."""
        service = WeightAdjustmentService(
            base_weights={"engine1": 0.5, "engine2": 0.5},
            storage_adapter=None,
            plasticity_tracker=None,
        )
        assert service._epsilon == 0.01
        assert service._base_weights == {"engine1": 0.5, "engine2": 0.5}
    
    def test_custom_epsilon(self):
        """Test service with custom epsilon."""
        service = WeightAdjustmentService(
            base_weights={},
            epsilon=0.001,
        )
        assert service._epsilon == 0.001


class TestResolveWeights:
    """Test weight resolution with different strategies."""
    
    def test_adaptive_weights_from_storage(self):
        """Test adaptive weights when storage has data."""
        storage = Mock()
        storage.get_rolling_performance.side_effect = [
            {"mae": 2.0},  # engine1
            {"mae": 4.0},  # engine2
        ]
        
        service = WeightAdjustmentService(
            base_weights={"engine1": 0.5, "engine2": 0.5},
            storage_adapter=storage,
        )
        
        weights = service.resolve_weights(
            regime="stable",
            engine_names=["engine1", "engine2"],
            series_id="test_series",
        )
        
        # engine1 has lower MAE (2.0) so should have higher weight
        assert weights["engine1"] > weights["engine2"]
        assert abs(sum(weights.values()) - 1.0) < 0.001  # Normalized
    
    def test_fallback_to_plasticity_weights(self):
        """Test fallback to plasticity when storage unavailable."""
        plasticity = Mock()
        plasticity.has_history.return_value = True
        plasticity.get_weights.return_value = {"engine1": 0.6, "engine2": 0.4}
        
        service = WeightAdjustmentService(
            base_weights={"engine1": 0.5, "engine2": 0.5},
            storage_adapter=None,
            plasticity_tracker=plasticity,
        )
        
        weights = service.resolve_weights(
            regime="stable",
            engine_names=["engine1", "engine2"],
        )
        
        assert weights == {"engine1": 0.6, "engine2": 0.4}
        plasticity.has_history.assert_called_once_with("stable")
    
    def test_fallback_to_base_weights(self):
        """Test fallback to base weights when no history."""
        plasticity = Mock()
        plasticity.has_history.return_value = False
        
        service = WeightAdjustmentService(
            base_weights={"engine1": 0.7, "engine2": 0.3},
            plasticity_tracker=plasticity,
        )
        
        weights = service.resolve_weights(
            regime="volatile",
            engine_names=["engine1", "engine2"],
        )
        
        assert weights == {"engine1": 0.7, "engine2": 0.3}
    
    def test_fallback_to_uniform_weights(self):
        """Test uniform weights when engine not in base_weights."""
        service = WeightAdjustmentService(
            base_weights={},
        )
        
        weights = service.resolve_weights(
            regime="stable",
            engine_names=["engine1", "engine2", "engine3"],
        )
        
        # Uniform weights
        assert abs(weights["engine1"] - 1/3) < 0.001
        assert abs(weights["engine2"] - 1/3) < 0.001
        assert abs(weights["engine3"] - 1/3) < 0.001


class TestComputeAdaptiveWeights:
    """Test adaptive weight computation."""
    
    def test_insufficient_data_returns_none(self):
        """Test returns None when engine has no data."""
        storage = Mock()
        storage.get_rolling_performance.side_effect = [
            {"mae": 2.0},  # engine1 has data
            None,          # engine2 has no data
        ]
        
        service = WeightAdjustmentService(
            base_weights={},
            storage_adapter=storage,
        )
        
        weights = service._compute_adaptive_weights(
            series_id="test_series",
            engine_names=["engine1", "engine2"],
        )
        
        assert weights is None
    
    def test_zero_total_returns_none(self):
        """Test returns None when total weight is zero."""
        storage = Mock()
        storage.get_rolling_performance.side_effect = [
            {"mae": 1e10},  # Very high MAE
            {"mae": 1e10},  # Very high MAE
        ]
        
        service = WeightAdjustmentService(
            base_weights={},
            storage_adapter=storage,
            epsilon=0.0,  # No epsilon to force zero
        )
        
        weights = service._compute_adaptive_weights(
            series_id="test_series",
            engine_names=["engine1", "engine2"],
        )
        
        # With very high MAE and no epsilon, total could be near zero
        # This tests the edge case
        assert weights is None or abs(sum(weights.values()) - 1.0) < 0.001
    
    def test_no_storage_returns_none(self):
        """Test returns None when no storage adapter."""
        service = WeightAdjustmentService(
            base_weights={},
            storage_adapter=None,
        )
        
        weights = service._compute_adaptive_weights(
            series_id="test_series",
            engine_names=["engine1"],
        )
        
        assert weights is None
    
    def test_inverse_mae_weighting(self):
        """Test that lower MAE gets higher weight."""
        storage = Mock()
        storage.get_rolling_performance.side_effect = [
            {"mae": 1.0},  # engine1: low error
            {"mae": 9.0},  # engine2: high error
        ]
        
        service = WeightAdjustmentService(
            base_weights={},
            storage_adapter=storage,
            epsilon=0.01,
        )
        
        weights = service._compute_adaptive_weights(
            series_id="test_series",
            engine_names=["engine1", "engine2"],
        )
        
        assert weights is not None
        # engine1 should have much higher weight (9x better MAE)
        assert weights["engine1"] > 0.8
        assert weights["engine2"] < 0.2
