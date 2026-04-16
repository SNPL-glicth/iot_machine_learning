"""Tests for SparseFusionLayer.

Verifica:
- Opera solo sobre k expertos (no todos)
- Normaliza pesos antes de fusionar
- Retorna Prediction del dominio
- SRP: solo fusión, no gating, no dispatch
"""

import pytest
import math

from domain.entities.prediction import Prediction
from domain.ports.expert_port import ExpertOutput
from ..sparse_fusion import SparseFusionLayer, FusionWeights


def create_expert_output(prediction: float, confidence: float, trend: str = "stable"):
    """Helper para crear ExpertOutput."""
    return ExpertOutput(
        prediction=prediction,
        confidence=confidence,
        trend=trend,
    )


class TestSparseFusionBasics:
    """Tests básicos de fusión."""
    
    def test_fuse_single_expert(self):
        """Fusiona output de un solo experto."""
        fusion = SparseFusionLayer()
        
        outputs = {
            "baseline": create_expert_output(10.0, 0.8),
        }
        weights = {"baseline": 1.0}
        
        prediction = fusion.fuse(outputs, weights)
        
        assert isinstance(prediction, Prediction)
        assert prediction.predicted_value == 10.0
        assert prediction.confidence == 0.8
    
    def test_fuse_two_experts(self):
        """Fusiona outputs de 2 expertos (k=2)."""
        fusion = SparseFusionLayer()
        
        outputs = {
            "baseline": create_expert_output(10.0, 0.8, "stable"),
            "taylor": create_expert_output(12.0, 0.7, "up"),
        }
        weights = {"baseline": 0.6, "taylor": 0.4}
        
        prediction = fusion.fuse(outputs, weights)
        
        # Media ponderada: 10*0.6 + 12*0.4 = 10.8
        assert prediction.predicted_value == 10.8
        # Confianza ponderada: 0.8*0.6 + 0.7*0.4 = 0.76
        assert prediction.confidence == 0.76
    
    def test_fuse_three_experts(self):
        """Fusiona outputs de 3 expertos (k=3)."""
        fusion = SparseFusionLayer()
        
        outputs = {
            "baseline": create_expert_output(10.0, 0.8),
            "statistical": create_expert_output(11.0, 0.75),
            "taylor": create_expert_output(12.0, 0.7),
        }
        weights = {
            "baseline": 0.5,
            "statistical": 0.3,
            "taylor": 0.2,
        }
        
        prediction = fusion.fuse(outputs, weights)
        
        # Media ponderada: 10*0.5 + 11*0.3 + 12*0.2 = 10.7
        assert prediction.predicted_value == 10.7


class TestWeightNormalization:
    """Tests de normalización de pesos."""
    
    def test_weights_normalized_to_sum_one(self):
        """Pesos normalizados suman 1.0."""
        fusion = SparseFusionLayer()
        
        weights = {"e1": 2.0, "e2": 3.0}  # Suma 5.0
        normalized = fusion._normalize_weights(weights)
        
        assert math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-9)
        assert normalized["e1"] == 0.4  # 2/5
        assert normalized["e2"] == 0.6  # 3/5
    
    def test_already_normalized_unchanged(self):
        """Pesos ya normalizados no cambian."""
        fusion = SparseFusionLayer()
        
        weights = {"e1": 0.7, "e2": 0.3}  # Ya suman 1.0
        normalized = fusion._normalize_weights(weights)
        
        assert normalized["e1"] == 0.7
        assert normalized["e2"] == 0.3
    
    def test_zero_weights_fallback_uniform(self):
        """Pesos cero caen a distribución uniforme."""
        fusion = SparseFusionLayer()
        
        weights = {"e1": 0.0, "e2": 0.0}
        normalized = fusion._normalize_weights(weights)
        
        assert normalized["e1"] == 0.5
        assert normalized["e2"] == 0.5


class TestSparsity:
    """Tests de operación sobre k expertos (no todos)."""
    
    def test_fusion_ignores_non_selected_experts(self):
        """Solo opera sobre los k expertos proporcionados."""
        fusion = SparseFusionLayer()
        
        # Solo 2 expertos en outputs (k=2)
        outputs = {
            "baseline": create_expert_output(10.0, 0.8),
            "taylor": create_expert_output(12.0, 0.7),
        }
        weights = {"baseline": 0.5, "taylor": 0.5}
        
        prediction = fusion.fuse(outputs, weights)
        
        # Metadata indica k=2
        assert prediction.metadata["fusion"]["sparsity_k"] == 2
    
    def test_k_experts_not_total_pool(self):
        """k puede ser menor que el pool total disponible."""
        fusion = SparseFusionLayer()
        
        # Solo usamos 1 experto de un pool hipotético de 3
        outputs = {"baseline": create_expert_output(10.0, 0.9)}
        weights = {"baseline": 1.0}
        
        prediction = fusion.fuse(outputs, weights)
        
        assert prediction.metadata["fusion"]["sparsity_k"] == 1


class TestPredictionOutput:
    """Tests de formato de salida."""
    
    def test_returns_domain_prediction(self):
        """Retorna Prediction del dominio."""
        fusion = SparseFusionLayer()
        
        outputs = {"e1": create_expert_output(10.0, 0.8)}
        weights = {"e1": 1.0}
        
        result = fusion.fuse(outputs, weights)
        
        assert isinstance(result, Prediction)
        assert hasattr(result, "predicted_value")
        assert hasattr(result, "confidence")
        assert hasattr(result, "trend")
        assert hasattr(result, "metadata")
    
    def test_metadata_contains_fusion_info(self):
        """Metadata contiene info de fusión."""
        fusion = SparseFusionLayer()
        
        outputs = {"e1": create_expert_output(10.0, 0.8)}
        weights = {"e1": 1.0}
        
        prediction = fusion.fuse(outputs, weights)
        
        fusion_meta = prediction.metadata["fusion"]
        assert "sparsity_k" in fusion_meta
        assert "weights_used" in fusion_meta
        assert "dominant_expert" in fusion_meta
        assert "uncertainty" in fusion_meta


class TestTrendSelection:
    """Tests de selección de tendencia."""
    
    def test_trend_from_dominant_expert(self):
        """Tendencia viene del experto con mayor peso."""
        fusion = SparseFusionLayer()
        
        outputs = {
            "baseline": create_expert_output(10.0, 0.8, "stable"),  # 80% peso
            "taylor": create_expert_output(12.0, 0.7, "up"),       # 20% peso
        }
        weights = {"baseline": 0.8, "taylor": 0.2}
        
        prediction = fusion.fuse(outputs, weights)
        
        # Tendencia del dominante (baseline = stable)
        assert prediction.trend == "stable"
    
    def test_trend_up_when_dominant_says_up(self):
        """Tendencia 'up' cuando el dominante lo indica."""
        fusion = SparseFusionLayer()
        
        outputs = {
            "baseline": create_expert_output(10.0, 0.8, "stable"),
            "taylor": create_expert_output(12.0, 0.9, "up"),  # Mayor confianza
        }
        weights = {"baseline": 0.3, "taylor": 0.7}  # Taylor dominante
        
        prediction = fusion.fuse(outputs, weights)
        
        assert prediction.trend == "up"


class TestUncertainty:
    """Tests de cálculo de incertidumbre."""
    
    def test_uncertainty_zero_for_single_expert(self):
        """Incertidumbre = 0 con un solo experto."""
        fusion = SparseFusionLayer()
        
        outputs = {"e1": create_expert_output(10.0, 0.8)}
        weights = {"e1": 1.0}
        
        prediction = fusion.fuse(outputs, weights)
        
        assert prediction.metadata["fusion"]["uncertainty"] == 0.0
    
    def test_uncertainty_with_disagreement(self):
        """Incertidumbre > 0 cuando expertos discrepan."""
        fusion = SparseFusionLayer()
        
        outputs = {
            "e1": create_expert_output(10.0, 0.8),
            "e2": create_expert_output(20.0, 0.8),  # Gran diferencia
        }
        weights = {"e1": 0.5, "e2": 0.5}
        
        prediction = fusion.fuse(outputs, weights)
        
        # Incertidumbre debe ser significativa
        uncertainty = prediction.metadata["fusion"]["uncertainty"]
        assert uncertainty > 4.0  # std de [10, 20] es ~5


class TestFusionWeights:
    """Tests de metadata de pesos."""
    
    def test_get_fusion_weights(self):
        """Obtiene pesos normalizados sin fusionar."""
        fusion = SparseFusionLayer()
        
        weights = {"e1": 0.7, "e2": 0.3}
        fusion_weights = fusion.get_fusion_weights(weights)
        
        assert isinstance(fusion_weights, FusionWeights)
        assert fusion_weights.dominant == "e1"
        assert fusion_weights.sparsity_k == 2
        assert math.isclose(sum(fusion_weights.normalized.values()), 1.0)


class TestSRP:
    """Tests de Single Responsibility Principle."""
    
    def test_no_gating_logic(self):
        """No contiene lógica de gating/routing."""
        fusion = SparseFusionLayer()
        
        # No tiene métodos de gating
        assert not hasattr(fusion, "route")
        assert not hasattr(fusion, "select_experts")
    
    def test_no_dispatch_logic(self):
        """No contiene lógica de dispatch/ejecución."""
        fusion = SparseFusionLayer()
        
        # No tiene métodos de dispatch
        assert not hasattr(fusion, "dispatch")
        assert not hasattr(fusion, "execute")
    
    def test_only_fusion_methods(self):
        """Solo tiene métodos relacionados a fusión."""
        fusion = SparseFusionLayer()
        
        methods = [m for m in dir(fusion) if not m.startswith("_")]
        
        # Solo fuse y get_fusion_weights son públicos
        assert "fuse" in methods
        assert "get_fusion_weights" in methods


class TestErrorHandling:
    """Tests de manejo de errores."""
    
    def test_empty_outputs_raises(self):
        """Outputs vacío lanza ValueError."""
        fusion = SparseFusionLayer()
        
        with pytest.raises(ValueError, match="No hay expert_outputs"):
            fusion.fuse({}, {"e1": 1.0})
    
    def test_empty_weights_raises(self):
        """Weights vacío lanza ValueError."""
        fusion = SparseFusionLayer()
        
        outputs = {"e1": create_expert_output(10.0, 0.8)}
        
        with pytest.raises(ValueError, match="No hay weights"):
            fusion.fuse(outputs, {})
