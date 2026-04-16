"""Tests para SparseFusionLayer.

Cobertura: fusión ponderada, cálculo de incertidumbre, normalización de pesos.
"""

import pytest
import math

from domain.ports.expert_port import ExpertOutput
from domain.entities.prediction import Prediction
from ..fusion.sparse_fusion import SparseFusionLayer, FusionWeights


class TestSparseFusionLayerBasics:
    """Tests básicos de fusión."""
    
    @pytest.fixture
    def fusion(self):
        return SparseFusionLayer()
    
    def test_fuse_single_expert(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="stable")
        }
        gating_probs = {"expert1": 1.0}
        
        result = fusion.fuse(outputs, gating_probs)
        
        assert isinstance(result, Prediction)
        assert result.predicted_value == 10.0
        assert result.confidence_score == 0.8
        assert result.trend == "stable"
    
    def test_fuse_two_experts(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="up"),
            "expert2": ExpertOutput(prediction=12.0, confidence=0.7, trend="up"),
        }
        gating_probs = {"expert1": 0.6, "expert2": 0.4}
        
        result = fusion.fuse(outputs, gating_probs)
        
        # Media ponderada: 10*0.6 + 12*0.4 = 10.8
        assert result.predicted_value == 10.8
        # Confianza ponderada: 0.8*0.6 + 0.7*0.4 = 0.76
        assert result.confidence_score == 0.76
        # Tendencia mayoritaria: up
        assert result.trend == "up"
        # Verificar metadata de fusión
        assert "fusion" in result.metadata
        assert result.metadata["fusion"]["sparsity_k"] == 2
    
    def test_fuse_different_trends(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="up"),
            "expert2": ExpertOutput(prediction=10.5, confidence=0.6, trend="down"),
        }
        gating_probs = {"expert1": 0.7, "expert2": 0.3}  # up tiene más peso
        
        result = fusion.fuse(outputs, gating_probs)
        
        # Tendencia: up (mayor peso)
        assert result.trend == "up"
    
    def test_fuse_renormalizes_weights(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="stable"),
            "expert2": ExpertOutput(prediction=12.0, confidence=0.7, trend="stable"),
        }
        # Pesos no normalizados (suman 0.5)
        gating_probs = {"expert1": 0.3, "expert2": 0.2}
        
        result = fusion.fuse(outputs, gating_probs)
        
        # Verificar weights en metadata - deben estar normalizados a 0.6 y 0.4
        assert result.metadata["fusion"]["weights_used"]["expert1"] == 0.6
        assert result.metadata["fusion"]["weights_used"]["expert2"] == 0.4


class TestFusionUncertainty:
    """Tests de cálculo de incertidumbre."""
    
    @pytest.fixture
    def fusion(self):
        return SparseFusionLayer()
    
    def test_uncertainty_with_agreement(self, fusion):
        # Expertos de acuerdo
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="stable"),
            "expert2": ExpertOutput(prediction=10.1, confidence=0.7, trend="stable"),
        }
        gating_probs = {"expert1": 0.5, "expert2": 0.5}
        
        result = fusion.fuse(outputs, gating_probs)
        
        # Baja incertidumbre (en metadata)
        uncertainty = result.metadata["fusion"]["uncertainty"]
        assert uncertainty < 0.1
    
    def test_uncertainty_with_disagreement(self, fusion):
        # Expertos en desacuerdo
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="stable"),
            "expert2": ExpertOutput(prediction=20.0, confidence=0.7, trend="stable"),
        }
        gating_probs = {"expert1": 0.5, "expert2": 0.5}
        
        result = fusion.fuse(outputs, gating_probs)
        
        # Alta incertidumbre (en metadata)
        uncertainty = result.metadata["fusion"]["uncertainty"]
        assert uncertainty > 4.0  # std ≈ 5
    
    def test_dominant_expert_is_highest_weight(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="stable"),
            "expert2": ExpertOutput(prediction=12.0, confidence=0.7, trend="stable"),
        }
        gating_probs = {"expert1": 0.8, "expert2": 0.2}
        
        result = fusion.fuse(outputs, gating_probs)
        
        # Dominant expert en metadata
        assert result.metadata["fusion"]["dominant_expert"] == "expert1"


class TestFusionErrors:
    """Tests de manejo de errores."""
    
    @pytest.fixture
    def fusion(self):
        return SparseFusionLayer()
    
    def test_empty_outputs_raises(self, fusion):
        with pytest.raises(ValueError) as exc_info:
            fusion.fuse({}, {})
        
        assert "No hay expert_outputs para fusionar" in str(exc_info.value)


class TestFusionNormalization:
    """Tests de normalización de pesos."""
    
    def test_normalize_weights_uniform_fallback(self):
        fusion = SparseFusionLayer()
        
        # Pesos que suman 0 (edge case)
        weights = {"e1": 0.0, "e2": 0.0}
        normalized = fusion._normalize_weights(weights)
        
        # Fallback a uniforme
        assert normalized["e1"] == 0.5
        assert normalized["e2"] == 0.5
    
    def test_normalize_weights_normal_case(self):
        fusion = SparseFusionLayer()
        
        weights = {"e1": 2.0, "e2": 3.0}  # Suma 5
        normalized = fusion._normalize_weights(weights)
        
        assert normalized["e1"] == 0.4
        assert normalized["e2"] == 0.6
