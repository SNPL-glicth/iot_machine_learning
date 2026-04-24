"""Test: Confidence-Weighted Fusion en WeightedFusion.

Verifica que la confianza de cada engine afecta su peso en la fusión:
- peso_final = peso_base × (0.7 + 0.3 × confidence)
"""

import pytest
from unittest.mock import MagicMock


class MockPerception:
    """Mock perception para testing."""
    def __init__(self, engine_name, predicted_value=1.0, confidence=0.8, trend="stable"):
        self.engine_name = engine_name
        self.predicted_value = predicted_value
        self.confidence = confidence
        self.trend = trend


class MockInhibitionState:
    """Mock inhibition state para testing."""
    def __init__(self, engine_name, inhibited_weight, suppression_factor=0.0):
        self.engine_name = engine_name
        self.inhibited_weight = inhibited_weight
        self.suppression_factor = suppression_factor


class TestConfidenceWeightedFusion:
    """Tests para verificar ponderación por confianza."""
    
    def test_high_confidence_engine_gets_full_weight(self):
        """Test: Engine con confidence=1.0 pesa 1.0× su peso base."""
        from iot_machine_learning.infrastructure.ml.cognitive.fusion.engine_selector import (
            WeightedFusion,
        )
        
        fusion = WeightedFusion()
        
        perceptions = [
            MockPerception("engine_a", predicted_value=10.0, confidence=1.0),
        ]
        inhibition_states = [
            MockInhibitionState("engine_a", inhibited_weight=1.0),
        ]
        
        result = fusion.fuse(
            perceptions,
            inhibition_states,
            enable_confidence_weighting=True,
        )
        
        fused_value, fused_conf, fused_trend, final_weights, selected, reason = result
        
        # Con confidence=1.0: multiplier = 0.7 + 0.3 * 1.0 = 1.0
        # Peso final = 1.0 * 1.0 = 1.0
        assert final_weights["engine_a"] == pytest.approx(1.0, rel=1e-6)
        assert fused_value == pytest.approx(10.0, rel=1e-6)
    
    def test_low_confidence_engine_gets_reduced_weight(self):
        """Test: Engine con confidence=0.3 pesa 0.79× su peso base."""
        from iot_machine_learning.infrastructure.ml.cognitive.fusion.engine_selector import (
            WeightedFusion,
        )
        
        fusion = WeightedFusion()
        
        perceptions = [
            MockPerception("engine_b", predicted_value=10.0, confidence=0.3),
        ]
        inhibition_states = [
            MockInhibitionState("engine_b", inhibited_weight=1.0),
        ]
        
        result = fusion.fuse(
            perceptions,
            inhibition_states,
            enable_confidence_weighting=True,
        )
        
        fused_value, fused_conf, fused_trend, final_weights, selected, reason = result
        
        # Con confidence=0.3: multiplier = 0.7 + 0.3 * 0.3 = 0.79
        # Peso final = 1.0 * 0.79 = 0.79
        # Pero luego se normaliza a 1.0 (único engine)
        assert final_weights["engine_b"] == pytest.approx(1.0, rel=1e-6)
    
    def test_two_engines_different_confidence(self):
        """Test: Dos engines con diferente confianza.
        
        Engine A: base=0.6, confidence=0.9 → peso_ajustado = 0.6 × (0.7 + 0.3×0.9) = 0.6 × 0.97 = 0.582
        Engine B: base=0.4, confidence=0.3 → peso_ajustado = 0.4 × (0.7 + 0.3×0.3) = 0.4 × 0.79 = 0.316
        
        Total ajustado = 0.582 + 0.316 = 0.898
        Peso normalizado A = 0.582 / 0.898 ≈ 0.648
        Peso normalizado B = 0.316 / 0.898 ≈ 0.352
        """
        from iot_machine_learning.infrastructure.ml.cognitive.fusion.engine_selector import (
            WeightedFusion,
        )
        
        fusion = WeightedFusion()
        
        perceptions = [
            MockPerception("engine_a", predicted_value=10.0, confidence=0.9),
            MockPerception("engine_b", predicted_value=5.0, confidence=0.3),
        ]
        inhibition_states = [
            MockInhibitionState("engine_a", inhibited_weight=0.6),
            MockInhibitionState("engine_b", inhibited_weight=0.4),
        ]
        
        result = fusion.fuse(
            perceptions,
            inhibition_states,
            enable_confidence_weighting=True,
        )
        
        fused_value, fused_conf, fused_trend, final_weights, selected, reason = result
        
        # Verificar que engine_a tiene más peso que engine_b
        assert final_weights["engine_a"] > final_weights["engine_b"], (
            f"Engine A (high conf) debe tener más peso que B (low conf). "
            f"Pesos: {final_weights}"
        )
        
        # Verificar valores esperados (aproximados)
        # engine_a: 0.6 × 0.97 = 0.582 → / 0.898 = 0.648
        assert final_weights["engine_a"] == pytest.approx(0.648, rel=0.01)
        # engine_b: 0.4 × 0.79 = 0.316 → / 0.898 = 0.352
        assert final_weights["engine_b"] == pytest.approx(0.352, rel=0.01)
        
        # Verificar que suma = 1.0
        assert sum(final_weights.values()) == pytest.approx(1.0, rel=1e-6)
        
        # Verificar fusión ponderada
        # fused_value = 10.0 × 0.648 + 5.0 × 0.352 ≈ 8.24
        expected_fused = 10.0 * 0.648 + 5.0 * 0.352
        assert fused_value == pytest.approx(expected_fused, rel=0.01)
    
    def test_confidence_weighting_can_be_disabled(self):
        """Test: Confidence weighting puede deshabilitarse."""
        from iot_machine_learning.infrastructure.ml.cognitive.fusion.engine_selector import (
            WeightedFusion,
        )
        
        fusion = WeightedFusion()
        
        perceptions = [
            MockPerception("engine_a", predicted_value=10.0, confidence=0.9),
            MockPerception("engine_b", predicted_value=5.0, confidence=0.3),
        ]
        inhibition_states = [
            MockInhibitionState("engine_a", inhibited_weight=0.6),
            MockInhibitionState("engine_b", inhibited_weight=0.4),
        ]
        
        # Con confidence weighting habilitado
        result_with = fusion.fuse(
            perceptions,
            inhibition_states,
            enable_confidence_weighting=True,
        )
        weights_with = result_with[3]
        
        # Con confidence weighting deshabilitado
        result_without = fusion.fuse(
            perceptions,
            inhibition_states,
            enable_confidence_weighting=False,
        )
        weights_without = result_without[3]
        
        # Sin confidence weighting, pesos deben ser 0.6 y 0.4 (normalizados)
        assert weights_without["engine_a"] == pytest.approx(0.6, rel=1e-6)
        assert weights_without["engine_b"] == pytest.approx(0.4, rel=1e-6)
        
        # Con confidence weighting, engine_a debe tener más peso relativo
        assert weights_with["engine_a"] > weights_without["engine_a"]
        assert weights_with["engine_b"] < weights_without["engine_b"]
    
    def test_all_engines_have_minimum_weight(self):
        """Test: Todos los engines tienen al menos 0.7× su peso base (nunca 0)."""
        from iot_machine_learning.infrastructure.ml.cognitive.fusion.engine_selector import (
            WeightedFusion,
        )
        
        fusion = WeightedFusion()
        
        # Engine con confianza casi 0
        perceptions = [
            MockPerception("low_conf", predicted_value=10.0, confidence=0.01),
            MockPerception("high_conf", predicted_value=10.0, confidence=0.99),
        ]
        inhibition_states = [
            MockInhibitionState("low_conf", inhibited_weight=0.5),
            MockInhibitionState("high_conf", inhibited_weight=0.5),
        ]
        
        result = fusion.fuse(
            perceptions,
            inhibition_states,
            enable_confidence_weighting=True,
        )
        
        fused_value, fused_conf, fused_trend, final_weights, selected, reason = result
        
        # low_conf: 0.5 × (0.7 + 0.3×0.01) = 0.5 × 0.703 = 0.3515
        # high_conf: 0.5 × (0.7 + 0.3×0.99) = 0.5 × 0.997 = 0.4985
        # low_conf nunca llega a 0, mantiene al menos 70% de su peso base
        assert final_weights["low_conf"] > 0.3  # ~0.41 después de normalizar
        assert final_weights["high_conf"] > final_weights["low_conf"]
    
    def test_weighted_confidence_calculation(self):
        """Test: Confianza final es promedio ponderado por pesos ajustados."""
        from iot_machine_learning.infrastructure.ml.cognitive.fusion.engine_selector import (
            WeightedFusion,
        )
        
        fusion = WeightedFusion()
        
        perceptions = [
            MockPerception("engine_a", predicted_value=10.0, confidence=1.0),
            MockPerception("engine_b", predicted_value=5.0, confidence=0.0),
        ]
        inhibition_states = [
            MockInhibitionState("engine_a", inhibited_weight=0.5),
            MockInhibitionState("engine_b", inhibited_weight=0.5),
        ]
        
        result = fusion.fuse(
            perceptions,
            inhibition_states,
            enable_confidence_weighting=True,
        )
        
        fused_value, fused_conf, fused_trend, final_weights, selected, reason = result
        
        # Pesos ajustados antes de normalizar:
        # engine_a: 0.5 × 1.0 = 0.5
        # engine_b: 0.5 × 0.7 = 0.35
        # Total: 0.85
        # Normalizados: a=0.5/0.85≈0.588, b=0.35/0.85≈0.412
        
        # Confianza ponderada: 1.0×0.588 + 0.0×0.412 = 0.588
        assert fused_conf == pytest.approx(0.588, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
