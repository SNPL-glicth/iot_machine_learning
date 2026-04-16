"""Tests para SparseFusionLayer.

Cobertura: fusión ponderada, cálculo de incertidumbre, detección de divergencia.
"""

import pytest
import math

from iot_machine_learning.domain.ports.expert_port import ExpertOutput
from ..fusion import SparseFusionLayer, FusedResult


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
        
        result = fusion.fuse(outputs, gating_probs, total_experts_available=1)
        
        assert result.value == 10.0
        assert result.confidence == 0.8
        assert result.computation_saved_pct == 0.0  # Solo 1 experto disponible
    
    def test_fuse_two_experts(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="up"),
            "expert2": ExpertOutput(prediction=12.0, confidence=0.7, trend="up"),
        }
        gating_probs = {"expert1": 0.6, "expert2": 0.4}
        
        result = fusion.fuse(outputs, gating_probs, total_experts_available=3)
        
        # Media ponderada: 10*0.6 + 12*0.4 = 10.8
        assert result.value == 10.8
        # Confianza ponderada: 0.8*0.6 + 0.7*0.4 = 0.76
        assert result.confidence == 0.76
        # Tendencia mayoritaria: up
        assert result.trend == "up"
        # Computación ahorrada: 1 - 2/3 = 33%
        assert result.computation_saved_pct == pytest.approx(33.33, rel=0.01)
    
    def test_fuse_different_trends(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="up"),
            "expert2": ExpertOutput(prediction=10.5, confidence=0.6, trend="down"),
        }
        gating_probs = {"expert1": 0.7, "expert2": 0.3}  # up tiene más peso
        
        result = fusion.fuse(outputs, gating_probs, total_experts_available=2)
        
        # Tendencia: up (mayor peso)
        assert result.trend == "up"
    
    def test_fuse_renormalizes_weights(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="stable"),
            "expert2": ExpertOutput(prediction=12.0, confidence=0.7, trend="stable"),
        }
        # Pesos no normalizados (suman 0.5)
        gating_probs = {"expert1": 0.3, "expert2": 0.2}
        
        result = fusion.fuse(outputs, gating_probs, total_experts_available=2)
        
        # Debe normalizar a 0.6 y 0.4
        assert result.weights["expert1"] == 0.6
        assert result.weights["expert2"] == 0.4


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
        
        result = fusion.fuse(outputs, gating_probs, total_experts_available=2)
        
        # Baja incertidumbre
        assert result.uncertainty < 0.1
    
    def test_uncertainty_with_disagreement(self, fusion):
        # Expertos en desacuerdo
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="stable"),
            "expert2": ExpertOutput(prediction=20.0, confidence=0.7, trend="stable"),
        }
        gating_probs = {"expert1": 0.5, "expert2": 0.5}
        
        result = fusion.fuse(outputs, gating_probs, total_experts_available=2)
        
        # Alta incertidumbre
        assert result.uncertainty > 4.0  # std ≈ 5
    
    def test_selected_expert_is_highest_weight(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="stable"),
            "expert2": ExpertOutput(prediction=12.0, confidence=0.7, trend="stable"),
        }
        gating_probs = {"expert1": 0.8, "expert2": 0.2}
        
        result = fusion.fuse(outputs, gating_probs, total_experts_available=2)
        
        assert result.selected_expert == "expert1"


class TestFusionDivergenceDetection:
    """Tests de detección de divergencia entre expertos."""
    
    @pytest.fixture
    def fusion(self):
        # Threshold bajo para detectar divergencias moderadas
        return SparseFusionLayer(max_disagreement_threshold=0.5)
    
    def test_no_divergence_low_threshold(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="stable"),
            "expert2": ExpertOutput(prediction=10.5, confidence=0.7, trend="stable"),
        }
        weights = {"expert1": 0.5, "expert2": 0.5}
        
        disagreement, is_significant = fusion.compute_expert_disagreement(outputs, weights)
        
        # 5% de diferencia no es significativa con threshold 10%
        assert not is_significant
    
    def test_divergence_detected(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="stable"),
            "expert2": ExpertOutput(prediction=30.0, confidence=0.7, trend="stable"),
        }
        weights = {"expert1": 0.5, "expert2": 0.5}
        
        disagreement, is_significant = fusion.compute_expert_disagreement(outputs, weights)
        
        # 200% diferencia (range=20, mean=20) -> disagreement = 1.0
        assert disagreement > 0.4
        assert is_significant
    
    def test_single_expert_no_disagreement(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=10.0, confidence=0.8, trend="stable"),
        }
        weights = {"expert1": 1.0}
        
        disagreement, is_significant = fusion.compute_expert_disagreement(outputs, weights)
        
        assert disagreement == 0.0
        assert not is_significant
    
    def test_should_alert_divergence(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=100.0, confidence=0.8, trend="stable"),
            "expert2": ExpertOutput(prediction=150.0, confidence=0.7, trend="stable"),
        }
        
        should_alert, explanation = fusion.should_alert_divergence(outputs, threshold_pct=10.0)
        
        # 20% de desviación (25/125) supera 10%
        assert should_alert
        assert "20." in explanation  # 20% o 20.0%
    
    def test_no_alert_within_threshold(self, fusion):
        outputs = {
            "expert1": ExpertOutput(prediction=100.0, confidence=0.8, trend="stable"),
            "expert2": ExpertOutput(prediction=105.0, confidence=0.7, trend="stable"),
        }
        
        should_alert, explanation = fusion.should_alert_divergence(outputs, threshold_pct=10.0)
        
        # 5% dentro de 10%
        assert not should_alert


class TestFusionErrors:
    """Tests de manejo de errores."""
    
    @pytest.fixture
    def fusion(self):
        return SparseFusionLayer()
    
    def test_empty_outputs_raises(self, fusion):
        with pytest.raises(ValueError) as exc_info:
            fusion.fuse({}, {}, total_experts_available=1)
        
        assert "No hay outputs para fusionar" in str(exc_info.value)


class TestFusedResultValidation:
    """Tests de validación de FusedResult."""
    
    def test_invalid_trend(self):
        with pytest.raises(ValueError) as exc_info:
            FusedResult(
                value=10.0,
                confidence=0.8,
                trend="invalid",  # No permitido
                weights={"e1": 1.0},
                uncertainty=0.0,
                selected_expert="e1",
                computation_saved_pct=0.0
            )
        
        assert "trend inválido" in str(exc_info.value)


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
