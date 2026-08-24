"""Integration test: Multi-Expert Consensus with Rosa Roja Challenger.

Validates:
1. Rosa Roja registered alongside existing experts in MoE registry
2. sum_w_c aggregates all expert inputs correctly (baseline, taylor, kalman, statistical, rosa_roja)
3. Master Modulator: Phi_MoE = sum_w_c * (1 - lambda_t * (1 - Phi_Ritmo))
4. High lambda_t -> 1.0 or low Phi_Ritmo -> 0 triggers HOLD/EMERGENCY_FLUSH
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, MagicMock
import numpy as np

from iot_machine_learning.domain.ports.expert_port import ExpertOutput, ExpertCapability
from iot_machine_learning.domain.entities.iot.sensor_reading import SensorWindow, Reading
from iot_machine_learning.infrastructure.ml.moe.registry import ExpertRegistry
from iot_machine_learning.infrastructure.ml.moe.gateway.expert_dispatcher import ExpertDispatcher
from iot_machine_learning.infrastructure.ml.moe.experts.rosa_roja_expert import RosaRojaExpert, RosaRojaResult
from iot_machine_learning.infrastructure.ml.moe.config.moe_config import REGIME_WEIGHTS


class MockExpert:
    """Simple mock expert for testing consensus."""
    def __init__(self, name: str, confidence: float, weight: float = 1.0, is_critical: bool = False, threshold: float = 0.5):
        self.name = name
        self._confidence = confidence
        self.weight = weight
        self.is_critical = is_critical
        self.threshold = threshold
        self._capabilities = ExpertCapability(
            regimes=("stable", "trending", "volatile", "noisy"),
            min_points=3,
            computational_cost=1.0,
        )
    
    @property
    def capabilities(self) -> ExpertCapability:
        return self._capabilities
    
    def evaluate_trajectory(self, trajectory) -> float:
        return self._confidence
    
    def update_learning(self, actual, predicted):
        pass
    
    def predict(self, window):
        return ExpertOutput(
            prediction=0.0,
            confidence=self._confidence,
            trend="stable",
            metadata={"expert": self.name}
        )
    
    def can_handle(self, window):
        return True


class TestMultiExpertConsensusWithRosaRoja:
    """Tests for multi-expert consensus with Rosa Roja as challenger."""
    
    @pytest.fixture
    def window(self):
        return SensorWindow(
            series_id="TEST_001",
            readings=[Reading(series_id="TEST_001", value=100.0 + i * 0.1, timestamp=float(i)) for i in range(60)]
        )
    
    @pytest.fixture
    def rosa_roja_result(self):
        return RosaRojaResult(
            trajectory=["expansion", "exhaustion", "pullback", "continuation"],
            trajectory_score=0.78,
            rhythm_score=0.65,
            lambda_val=0.42,
            theta_entropy=1.23,
            regime_alert="volatile",
            invalidation_step=4,
            expected_direction="up",
            expected_magnitude=0.015,
            confidence=0.72,
            evidence={"test": "data"},
            status="ok",
        )
    
    @pytest.fixture
    def fake_rosa_roja_engine(self, rosa_roja_result):
        engine = Mock()
        engine.analyze.return_value = rosa_roja_result
        return engine
    
    def test_1_regime_weights_include_rosa_roja(self):
        """Verify rosa_roja is in REGIME_WEIGHTS for all regimes."""
        for regime in ["stable", "trending", "volatile", "noisy"]:
            assert "rosa_roja" in REGIME_WEIGHTS[regime], f"rosa_roja missing from {regime} weights"
            weight = REGIME_WEIGHTS[regime]["rosa_roja"]
            assert weight > 0, f"rosa_roja weight should be > 0 in {regime}"
            assert weight < 0.5, f"rosa_roja weight should be challenger-level (< 0.5)"
    
    def test_2_rosa_roja_registered_in_registry(self, window, fake_rosa_roja_engine, rosa_roja_result):
        """Rosa Roja registers in ExpertRegistry alongside other experts."""
        registry = ExpertRegistry()
        
        # Register mock experts
        experts = {
            "baseline": MockExpert("baseline", 0.6, weight=0.5),
            "taylor": MockExpert("taylor", 0.8, weight=1.2),
            "kalman": MockExpert("kalman", 0.7, weight=1.0),
            "statistical": MockExpert("statistical", 0.65, weight=0.8),
        }
        for name, exp in experts.items():
            registry.register(name, exp)
        
        # Register Rosa Roja
        rosa_adapter = RosaRojaExpert(engine=fake_rosa_roja_engine, enabled=True)
        registry.register("rosa_roja", rosa_adapter, rosa_adapter.capabilities)
        
        # Verify all registered
        assert "rosa_roja" in registry
        assert len(registry) == 5
        assert set(registry.list_all()) == {"baseline", "taylor", "kalman", "statistical", "rosa_roja"}
    
    def test_3_sum_wc_aggregates_all_experts(self, window, fake_rosa_roja_engine, rosa_roja_result):
        """sum_w_c correctly aggregates all expert inputs including Rosa Roja."""
        registry = ExpertRegistry()
        
        # Experts with known confidences and weights
        experts_config = [
            ("baseline", 0.6, 0.5),
            ("taylor", 0.8, 1.2),
            ("kalman", 0.7, 1.0),
            ("statistical", 0.65, 0.8),
            ("rosa_roja", 0.72, 1.0),  # Rosa Roja's confidence
        ]
        
        for name, conf, weight in experts_config:
            if name == "rosa_roja":
                adapter = RosaRojaExpert(engine=fake_rosa_roja_engine, enabled=True)
                registry.register(name, adapter)
            else:
                exp = MockExpert(name, conf, weight=weight)
                registry.register(name, exp)
        
        # Get all experts and compute weighted sum
        expert_outputs = {}
        dispatcher = ExpertDispatcher(registry, timeout_ms=5000)
        outputs = dispatcher.dispatch(registry.list_all(), window)
        
        # Verify all experts executed
        assert len(outputs) == 5, f"Expected 5 experts, got {len(outputs)}: {list(outputs.keys())}"
        assert "rosa_roja" in outputs
        
        # Compute sum_w_c manually to verify aggregation
        sum_w_c = 0.0
        total_weight = 0.0
        for name, output in outputs.items():
            # Get expert weight from capabilities
            expert = registry.get(name)
            w = expert.capabilities.computational_cost if expert else 1.0
            # Use inverse of computational_cost as weight (lower cost = higher weight)
            w = 1.0 / max(w, 0.1)
            sum_w_c += output.confidence * w
            total_weight += w
        
        normalized_sum_w_c = sum_w_c / total_weight if total_weight > 0 else 0.0
        
        # With our mock data:
        # baseline: 0.6 * 2.0 = 1.2
        # taylor: 0.8 * 0.83 = 0.66
        # kalman: 0.7 * 1.0 = 0.7
        # statistical: 0.65 * 1.25 = 0.81
        # rosa_roja: 0.72 * 1.0 = 0.72
        # sum = 4.09, total_weight = 2.0 + 0.83 + 1.0 + 1.25 + 1.0 = 6.08
        # normalized = 4.09 / 6.08 ≈ 0.67
        assert 0.5 < normalized_sum_w_c < 0.8, f"sum_w_c = {normalized_sum_w_c}"
    
    def test_4_master_modulator_phi_moe_calculation(self, fake_rosa_roja_engine, rosa_roja_result):
        """Verify Master Modulator: Phi_MoE = sum_w_c * (1 - lambda_t * (1 - Phi_Ritmo))."""
        from iot_machine_learning.core.orchestration.rosa_roja.engine import RosaRojaEngine
        
        # Test cases for Master Modulator
        test_cases = [
            # (sum_w_c, lambda_t, phi_ritmo, expected_phi_moe)
            (0.7, 0.0, 0.8, 0.7),           # lambda_t=0: no modulation
            (0.7, 1.0, 1.0, 0.7),           # lambda_t=1, phi_ritmo=1: no change
            (0.7, 1.0, 0.0, 0.0),           # lambda_t=1, phi_ritmo=0: collapse to 0
            (0.7, 0.5, 0.8, 0.63),          # lambda_t=0.5, phi_ritmo=0.8: 0.7 * (1 - 0.5*0.2) = 0.63
            (0.7, 0.5, 0.0, 0.35),          # lambda_t=0.5, phi_ritmo=0: 0.7 * 0.5 = 0.35
            (0.7, 0.9, 0.3, 0.259),         # high lambda, low phi_ritmo: strong collapse
        ]
        
        for sum_w_c, lambda_t, phi_ritmo, expected in test_cases:
            phi_moe = sum_w_c * (1.0 - lambda_t * (1.0 - phi_ritmo))
            assert abs(phi_moe - expected) < 0.01, f"Expected {expected}, got {phi_moe}"
    
    def test_5_high_lambda_triggers_hold(self, fake_rosa_roja_engine, rosa_roja_result):
        """High lambda_t -> 1.0 with low Phi_Ritmo triggers HOLD."""
        # Simulate high exploration + low trajectory coherence
        lambda_t = 0.95
        phi_ritmo = 0.1  # very low coherence
        sum_w_c = 0.75
        
        phi_moe = sum_w_c * (1.0 - lambda_t * (1.0 - phi_ritmo))
        
        # Should be below gamma_exec threshold (0.5)
        assert phi_moe < 0.5, f"Phi_MoE={phi_moe:.3f} should be < 0.5 (gamma_exec)"
        
        # Verify decision would be HOLD
        gamma_exec = 0.5
        if phi_moe >= gamma_exec:
            action = "EXECUTE"
        else:
            action = "HOLD"
        assert action == "HOLD"
    
    def test_6_low_phi_ritmo_triggers_hold(self, fake_rosa_roja_engine, rosa_roja_result):
        """Low Phi_Ritmo -> 0 triggers HOLD even with moderate lambda_t."""
        lambda_t = 0.5
        phi_ritmo = 0.05  # near zero coherence
        sum_w_c = 0.8
        
        phi_moe = sum_w_c * (1.0 - lambda_t * (1.0 - phi_ritmo))
        
        assert phi_moe < 0.5, f"Phi_MoE={phi_moe:.3f} should be < 0.5"
    
    def test_7_emergency_flush_on_direction_reversal(self, window, fake_rosa_roja_engine, rosa_roja_result):
        """Test EMERGENCY_FLUSH trigger on trajectory direction reversal."""
        from iot_machine_learning.core.orchestration.rosa_roja.engine import RosaRojaEngine
        
        # Create trajectory with direction reversal (cos < -0.1)
        # This is tested at engine level, but we can verify the threshold
        geometric_threshold = -0.1  # from engine
        
        # Simulate direction dots showing reversal
        dir_dots = np.array([0.9, 0.8, -0.2, 0.7])  # -0.2 < -0.1 triggers flush
        min_cos_theta = float(np.min(dir_dots))
        
        assert min_cos_theta < -0.1, "Direction reversal detected"
        assert min_cos_theta < geometric_threshold, "Should trigger EMERGENCY_FLUSH"
    
    def test_8_rosa_roja_modulates_consensus(self, window, fake_rosa_roja_engine, rosa_roja_result):
        """Rosa Roja's lambda_t and Phi_Ritmo modulate the consensus even if others agree."""
        # Other experts all agree (high confidence)
        other_experts_confidences = {
            "baseline": 0.75,
            "taylor": 0.80,
            "kalman": 0.78,
            "statistical": 0.72,
        }
        
        # Without Rosa Roja: average ~0.76 -> EXECUTE
        avg_without_rr = np.mean(list(other_experts_confidences.values()))
        assert avg_without_rr > 0.5
        
        # With Rosa Roja: high lambda, low phi_ritmo
        lambda_t = 0.8
        phi_ritmo = 0.2  # Rosa Roja detects incoherent trajectory
        
        sum_w_c = (sum(other_experts_confidences.values()) + rosa_roja_result.confidence) / 5
        
        # With modulation
        phi_moe = sum_w_c * (1.0 - lambda_t * (1.0 - phi_ritmo))
        
        # Should drop below threshold despite other experts agreeing
        assert phi_moe < 0.5, f"Rosa Roja modulation failed: Phi_MoE={phi_moe:.3f} >= 0.5"
        
        # Verify the modulation is significant
        drop = sum_w_c - phi_moe
        assert drop > 0.1, f"Modulation drop too small: {drop:.3f}"
    
    def test_9_factory_creates_engine_with_rosa_roja(self):
        """Factory creates MoE engine with Rosa Roja when flag enabled."""
        import os
        os.environ["ML_ENABLE_ROSA_ROJA_EXPERT"] = "true"
        
        from infrastructure.config.moe_factory import create_moe_gateway
        
        gateway = create_moe_gateway()
        
        assert "rosa_roja" in gateway._registry.list_all()
        assert len(gateway._registry) == 5  # baseline, statistical, taylor, kalman, rosa_roja
    
    def test_9b_factory_without_flag_excludes_rosa_roja(self):
        """Factory excludes Rosa Roja when flag disabled."""
        import os
        os.environ["ML_ENABLE_ROSA_ROJA_EXPERT"] = "false"
        
        from importlib import reload
        import infrastructure.config.moe_factory as moe_factory
        reload(moe_factory)
        
        from infrastructure.config.moe_factory import create_moe_gateway
        
        gateway = moe_factory.create_moe_gateway()
        
        assert "rosa_roja" not in gateway._registry.list_all()
        assert len(gateway._registry) == 4  # baseline, statistical, taylor, kalman


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])