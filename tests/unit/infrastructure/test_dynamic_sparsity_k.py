"""Tests for dynamic sparsity k based on gating entropy.

Verifies that:
1. High entropy (uncertain) → k=3
2. Low entropy (confident) → k=1
3. Medium entropy → k=2
4. Edge cases handled (0 experts, 1 expert)
5. Metadata includes dynamic_k and entropy_normalized
"""

import math
import pytest
from unittest.mock import MagicMock


class TestDynamicSparsityK:
    """Test dynamic sparsity k calculation based on entropy."""
    
    def test_high_entropy_returns_k3(self):
        """Test that uniform distribution (high entropy) returns k=3."""
        from iot_machine_learning.infrastructure.ml.moe.gateway.moe_gateway import (
            MoEGateway,
        )
        
        # Create mock gateway
        mock_registry = MagicMock()
        mock_gating = MagicMock()
        mock_fusion = MagicMock()
        mock_fallback = MagicMock()
        
        gateway = MoEGateway(
            registry=mock_registry,
            gating=mock_gating,
            fusion=mock_fusion,
            fallback_engine=mock_fallback,
        )
        
        # Uniform distribution (max entropy) with 3 experts
        gating_probs = {"expert_a": 0.33, "expert_b": 0.33, "expert_c": 0.34}
        
        dynamic_k, entropy_norm = gateway._compute_dynamic_k(gating_probs)
        
        # Uniform distribution should have high normalized entropy (>0.7)
        assert entropy_norm > 0.7, f"Expected high entropy, got {entropy_norm}"
        assert dynamic_k == 3, f"Expected k=3 for high entropy, got k={dynamic_k}"
    
    def test_low_entropy_returns_k1(self):
        """Test that dominant expert (low entropy) returns k=1."""
        from iot_machine_learning.infrastructure.ml.moe.gateway.moe_gateway import (
            MoEGateway,
        )
        
        mock_registry = MagicMock()
        mock_gating = MagicMock()
        mock_fusion = MagicMock()
        mock_fallback = MagicMock()
        
        gateway = MoEGateway(
            registry=mock_registry,
            gating=mock_gating,
            fusion=mock_fusion,
            fallback_engine=mock_fallback,
        )
        
        # Dominant expert (low entropy) - one expert at 95%
        gating_probs = {"expert_a": 0.95, "expert_b": 0.03, "expert_c": 0.02}
        
        dynamic_k, entropy_norm = gateway._compute_dynamic_k(gating_probs)
        
        # Dominant distribution should have low normalized entropy (<0.3)
        assert entropy_norm < 0.3, f"Expected low entropy, got {entropy_norm}"
        assert dynamic_k == 1, f"Expected k=1 for low entropy, got k={dynamic_k}"
    
    def test_medium_entropy_returns_k2(self):
        """Test that medium entropy returns k=2."""
        from iot_machine_learning.infrastructure.ml.moe.gateway.moe_gateway import (
            MoEGateway,
        )
        
        mock_registry = MagicMock()
        mock_gating = MagicMock()
        mock_fusion = MagicMock()
        mock_fallback = MagicMock()
        
        gateway = MoEGateway(
            registry=mock_registry,
            gating=mock_gating,
            fusion=mock_fusion,
            fallback_engine=mock_fallback,
        )
        
        # Medium entropy - one expert clearly dominant but not overwhelming
        # 0.80/0.15/0.05 should give normalized entropy in medium range (~0.5)
        gating_probs = {"expert_a": 0.80, "expert_b": 0.15, "expert_c": 0.05}
        
        dynamic_k, entropy_norm = gateway._compute_dynamic_k(gating_probs)
        
        # Should have medium normalized entropy (0.3-0.7)
        # Note: entropy > 0.7 triggers k=3, so we check for actual k=2 result
        assert dynamic_k == 2, f"Expected k=2 for medium entropy, got k={dynamic_k} with entropy={entropy_norm}"
    
    def test_zero_experts_returns_k1(self):
        """Test edge case with zero experts."""
        from iot_machine_learning.infrastructure.ml.moe.gateway.moe_gateway import (
            MoEGateway,
        )
        
        mock_registry = MagicMock()
        mock_gating = MagicMock()
        mock_fusion = MagicMock()
        mock_fallback = MagicMock()
        
        gateway = MoEGateway(
            registry=mock_registry,
            gating=mock_gating,
            fusion=mock_fusion,
            fallback_engine=mock_fallback,
        )
        
        gating_probs = {}
        
        dynamic_k, entropy_norm = gateway._compute_dynamic_k(gating_probs)
        
        assert dynamic_k == 1, f"Expected k=1 for empty distribution, got k={dynamic_k}"
        assert entropy_norm == 0.0
    
    def test_single_expert_returns_k1(self):
        """Test edge case with single expert."""
        from iot_machine_learning.infrastructure.ml.moe.gateway.moe_gateway import (
            MoEGateway,
        )
        
        mock_registry = MagicMock()
        mock_gating = MagicMock()
        mock_fusion = MagicMock()
        mock_fallback = MagicMock()
        
        gateway = MoEGateway(
            registry=mock_registry,
            gating=mock_gating,
            fusion=mock_fusion,
            fallback_engine=mock_fallback,
        )
        
        gating_probs = {"expert_a": 1.0}
        
        dynamic_k, entropy_norm = gateway._compute_dynamic_k(gating_probs)
        
        assert dynamic_k == 1, f"Expected k=1 for single expert, got k={dynamic_k}"
        assert entropy_norm == 0.0
    
    def test_k_respects_num_experts_available(self):
        """Test that k is capped by number of available experts."""
        from iot_machine_learning.infrastructure.ml.moe.gateway.moe_gateway import (
            MoEGateway,
        )
        
        mock_registry = MagicMock()
        mock_gating = MagicMock()
        mock_fusion = MagicMock()
        mock_fallback = MagicMock()
        
        gateway = MoEGateway(
            registry=mock_registry,
            gating=mock_gating,
            fusion=mock_fusion,
            fallback_engine=mock_fallback,
        )
        
        # Only 2 experts available but high entropy would suggest k=3
        gating_probs = {"expert_a": 0.5, "expert_b": 0.5}
        
        dynamic_k, entropy_norm = gateway._compute_dynamic_k(gating_probs)
        
        # k should be capped at num_experts (2)
        assert dynamic_k <= 2, f"Expected k<=2 for 2 experts, got k={dynamic_k}"
    
    def test_entropy_calculation_correctness(self):
        """Test entropy calculation is mathematically correct."""
        from iot_machine_learning.infrastructure.ml.moe.gateway.moe_gateway import (
            MoEGateway,
        )
        
        mock_registry = MagicMock()
        mock_gating = MagicMock()
        mock_fusion = MagicMock()
        mock_fallback = MagicMock()
        
        gateway = MoEGateway(
            registry=mock_registry,
            gating=mock_gating,
            fusion=mock_fusion,
            fallback_engine=mock_fallback,
        )
        
        # Uniform distribution of 3 experts
        gating_probs = {"a": 1/3, "b": 1/3, "c": 1/3}
        
        dynamic_k, entropy_norm = gateway._compute_dynamic_k(gating_probs)
        
        # For uniform distribution, normalized entropy should be close to 1.0
        # H_max = log(3) ≈ 1.0986
        # H_actual = log(3) ≈ 1.0986
        # normalized = H_actual / H_max ≈ 1.0
        assert abs(entropy_norm - 1.0) < 0.1, f"Expected normalized entropy ≈1.0, got {entropy_norm}"
    
    def test_metadata_includes_dynamic_k_and_entropy(self):
        """Test that MoEMetadata includes dynamic_k and entropy_normalized."""
        from iot_machine_learning.infrastructure.ml.moe.gateway.prediction_enricher import (
            MoEMetadata,
        )
        
        metadata = MoEMetadata(
            selected_experts=["expert_a"],
            sparsity_k=1,
            gating_probs={"expert_a": 1.0},
            fusion_weights={"expert_a": 1.0},
            dominant_expert="expert_a",
            total_latency_ms=10.0,
            moe_enabled=True,
            dynamic_k=1,
            entropy_normalized=0.15,
        )
        
        assert metadata.dynamic_k == 1
        assert metadata.entropy_normalized == 0.15
    
    def test_moemetadata_defaults_none_for_optional_fields(self):
        """Test that MoEMetadata defaults for optional fields are None."""
        from iot_machine_learning.infrastructure.ml.moe.gateway.prediction_enricher import (
            MoEMetadata,
        )
        
        metadata = MoEMetadata(
            selected_experts=["expert_a"],
            sparsity_k=1,
            gating_probs={"expert_a": 1.0},
            fusion_weights={"expert_a": 1.0},
            dominant_expert="expert_a",
            total_latency_ms=10.0,
            moe_enabled=True,
        )
        
        # Optional fields should default to None
        assert metadata.dynamic_k is None
        assert metadata.entropy_normalized is None
    
    def test_enricher_includes_dynamic_fields_in_output(self):
        """Test that enricher includes dynamic fields in metadata output."""
        from iot_machine_learning.infrastructure.ml.moe.gateway.prediction_enricher import (
            MoEMetadata,
            PredictionEnricher,
        )
        from iot_machine_learning.domain.entities.prediction import Prediction
        from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
        
        # Create mock objects
        mock_prediction = Prediction(
            series_id="test",
            predicted_value=10.0,
            confidence_score=0.8,
            trend="stable",
            engine_name="moe",
            metadata={},
        )
        
        mock_window = MagicMock()
        mock_window.sensor_id = 123
        
        metadata = MoEMetadata(
            selected_experts=["expert_a", "expert_b"],
            sparsity_k=2,
            gating_probs={"expert_a": 0.6, "expert_b": 0.4},
            fusion_weights={"expert_a": 0.6, "expert_b": 0.4},
            dominant_expert="expert_a",
            total_latency_ms=15.5,
            moe_enabled=True,
            dynamic_k=2,
            entropy_normalized=0.6734,
        )
        
        enricher = PredictionEnricher()
        enriched = enricher.enrich(mock_prediction, metadata, mock_window)
        
        # Verify metadata includes dynamic fields
        moe_meta = enriched.metadata["moe"]
        assert "dynamic_k" in moe_meta
        assert "entropy_normalized" in moe_meta
        assert moe_meta["dynamic_k"] == 2
        assert moe_meta["entropy_normalized"] == 0.6734  # Rounded to 4 decimals


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
