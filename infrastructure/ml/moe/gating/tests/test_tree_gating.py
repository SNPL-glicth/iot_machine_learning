"""Tests para TreeGatingNetwork.

Verifica:
- Entrenamiento con XGBoost
- Routing con modelo entrenado
- Explicabilidad SHAP
- Persistencia de modelo
"""

import pytest
import tempfile
from pathlib import Path

from domain.model.context_vector import ContextVector
from ..tree_gating import TreeGatingNetwork, TreeRoutingExplanation


@pytest.fixture
def sample_contexts():
    """Contextos de ejemplo para entrenamiento."""
    return [
        ContextVector(regime="stable", domain="iot", n_points=5, signal_features={}),
        ContextVector(regime="stable", domain="iot", n_points=6, signal_features={}),
        ContextVector(regime="volatile", domain="iot", n_points=10, signal_features={}),
        ContextVector(regime="volatile", domain="iot", n_points=12, signal_features={}),
        ContextVector(regime="trending", domain="iot", n_points=8, signal_features={}),
        ContextVector(regime="trending", domain="iot", n_points=9, signal_features={}),
    ]


@pytest.fixture
def sample_labels():
    """Labels de expertos para entrenamiento."""
    return ["baseline", "baseline", "taylor", "taylor", "statistical", "statistical"]


class TestTreeGatingNetwork:
    """Tests básicos de TreeGatingNetwork."""
    
    def test_fallback_when_untrained(self):
        """Cuando no entrenado, usa fallback uniforme."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gating = TreeGatingNetwork(
                expert_ids=["baseline", "taylor"],
                artifact_path=tmpdir,
            )
            
            ctx = ContextVector(regime="stable", domain="iot", n_points=5, signal_features={})
            probs = gating.route(ctx)
            
            # Fallback uniforme
            assert probs.probabilities["baseline"] == 0.5
            assert probs.probabilities["taylor"] == 0.5
    
    def test_train_returns_metrics(self, sample_contexts, sample_labels):
        """Entrenamiento retorna métricas."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gating = TreeGatingNetwork(
                expert_ids=["baseline", "taylor", "statistical"],
                artifact_path=tmpdir,
            )
            
            metrics = gating.train(sample_contexts, sample_labels)
            
            assert "n_samples" in metrics
            assert metrics["n_samples"] == 6
    
    def test_route_after_training(self, sample_contexts, sample_labels):
        """Después de entrenar, rutea con probabilidades."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gating = TreeGatingNetwork(
                expert_ids=["baseline", "taylor", "statistical"],
                artifact_path=tmpdir,
            )
            
            gating.train(sample_contexts, sample_labels)
            
            ctx = ContextVector(regime="stable", domain="iot", n_points=5, signal_features={})
            probs = gating.route(ctx)
            
            # Debe tener probabilidades para todos los expertos
            assert len(probs.probabilities) == 3
            assert sum(probs.probabilities.values()) > 0.99  # ~1.0
    
    def test_explain_route_returns_explanation(self, sample_contexts, sample_labels):
        """explain_route retorna TreeRoutingExplanation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            gating = TreeGatingNetwork(
                expert_ids=["baseline", "taylor", "statistical"],
                artifact_path=tmpdir,
            )
            
            gating.train(sample_contexts, sample_labels)
            
            ctx = ContextVector(regime="stable", domain="iot", n_points=5, signal_features={})
            explanation = gating.explain_route(ctx)
            
            if explanation is not None:  # SHAP puede no estar instalado
                assert isinstance(explanation, TreeRoutingExplanation)
                assert hasattr(explanation, 'expected_expert')
                assert hasattr(explanation, 'confidence')
    
    def test_model_persistence(self, sample_contexts, sample_labels):
        """Modelo se guarda y puede cargarse."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Crear y entrenar
            gating1 = TreeGatingNetwork(
                expert_ids=["baseline", "taylor"],
                artifact_path=tmpdir,
            )
            gating1.train(sample_contexts[:4], sample_labels[:4])
            
            # Crear nueva instancia (debe cargar modelo)
            gating2 = TreeGatingNetwork(
                expert_ids=["baseline", "taylor"],
                artifact_path=tmpdir,
            )
            
            # Debe poder rutear
            ctx = ContextVector(regime="stable", domain="iot", n_points=5, signal_features={})
            probs = gating2.route(ctx)
            
            assert len(probs.probabilities) == 2


class TestTreeRoutingExplanation:
    """Tests de TreeRoutingExplanation."""
    
    def test_explanation_structure(self):
        """Estructura correcta de explicación."""
        exp = TreeRoutingExplanation(
            top_features=[{"feature": "n_points", "importance": 0.5}],
            shap_values={"n_points": 0.3, "regime": 0.2},
            expected_expert="taylor",
            confidence=0.85,
        )
        
        assert exp.expected_expert == "taylor"
        assert exp.confidence == 0.85
        assert len(exp.top_features) == 1
