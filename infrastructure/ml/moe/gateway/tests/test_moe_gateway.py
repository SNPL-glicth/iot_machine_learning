"""Tests de integración para MoEGateway.

Verifica flujo completo y feature flag ML_MOE_ENABLED.
"""

import pytest
from dataclasses import dataclass
from typing import Dict

from domain.ports.prediction_port import PredictionPort
from domain.ports.expert_port import ExpertPort, ExpertOutput, ExpertCapability
from domain.entities.prediction import Prediction
from domain.entities.sensor_reading import SensorWindow
from domain.model.context_vector import ContextVector
from infrastructure.ml.moe.registry.expert_registry import ExpertRegistry
from infrastructure.ml.moe.gating.base import GatingNetwork, GatingProbs
from infrastructure.ml.moe.fusion.sparse_fusion import SparseFusionLayer

from ..moe_gateway import MoEGateway


@dataclass
class MockExpert(ExpertPort):
    """Mock experto para testing."""
    
    _name: str
    _prediction: float
    _confidence: float = 0.8
    _trend: str = "stable"
    _caps: ExpertCapability = None
    
    def __post_init__(self):
        if self._caps is None:
            self._caps = ExpertCapability(regimes=("stable", "volatile"))
    
    @property
    def name(self) -> str: return self._name
    
    @property
    def capabilities(self) -> ExpertCapability: return self._caps
    
    def predict(self, window: SensorWindow) -> ExpertOutput:
        return ExpertOutput(prediction=self._prediction, confidence=self._confidence, trend=self._trend)
    
    def can_handle(self, window: SensorWindow) -> bool:
        return len(window.readings) >= 3


@dataclass
class MockFallbackEngine(PredictionPort):
    """Mock engine fallback."""
    
    @property
    def name(self) -> str: return "fallback"
    
    def predict(self, window: SensorWindow) -> Prediction:
        return Prediction(series_id="test", predicted_value=50.0, confidence_score=0.5, trend="stable", engine_name="fallback", metadata={"source": "fallback"})
    
    def can_handle(self, n_points: int) -> bool: return n_points >= 1


@dataclass
class MockGatingNetwork(GatingNetwork):
    """Mock gating para testing controlado."""
    
    _probs: Dict[str, float]
    
    def route(self, context: ContextVector) -> GatingProbs:
        return GatingProbs(probabilities=self._probs, entropy=0.5, top_expert=max(self._probs.items(), key=lambda x: x[1])[0])
    
    def explain(self, context: ContextVector, probs: GatingProbs) -> str:
        return "Mock explanation"


def create_test_window(values=None) -> SensorWindow:
    """Helper para crear ventana de test."""
    from domain.entities.iot.sensor_reading import Reading
    if values is None:
        values = [10.0, 11.0, 12.0, 13.0, 14.0]
    
    readings = [Reading(series_id="test", value=v, timestamp=1000.0 + i) for i, v in enumerate(values)]
    return SensorWindow(series_id="test", readings=readings)


class TestMoEGatewayBasic:
    """Tests de flujo básico."""
    
    def test_predict_returns_prediction(self):
        """predict() retorna Prediction del dominio."""
        registry = ExpertRegistry()
        baseline = MockExpert("baseline", _prediction=10.0, _confidence=0.9)
        taylor = MockExpert("taylor", _prediction=20.0, _confidence=0.7)
        registry.register("baseline", baseline, baseline.capabilities)
        registry.register("taylor", taylor, taylor.capabilities)
        
        gating = MockGatingNetwork({"baseline": 0.6, "taylor": 0.4})
        gateway = MoEGateway(registry=registry, gating=gating, fusion=SparseFusionLayer(), fallback_engine=MockFallbackEngine(), sparsity_k=2)
        
        result = gateway.predict(create_test_window())
        
        assert isinstance(result, Prediction)
        assert result.predicted_value == 14.0  # 10*0.6 + 20*0.4


class TestFeatureFlag:
    """Tests de feature flag."""
    
    def test_moe_disabled_uses_fallback(self):
        """Si moe_enabled=false, usa fallback."""
        gateway = MoEGateway(
            registry=ExpertRegistry(),
            gating=MockGatingNetwork({"baseline": 1.0}),
            fusion=SparseFusionLayer(),
            fallback_engine=MockFallbackEngine(),
            moe_enabled=False,
        )
        
        result = gateway.predict(create_test_window())
        
        assert result.predicted_value == 50.0  # Valor del fallback
        assert result.metadata.get("source") == "fallback"
    
    def test_moe_enabled_executes_moe(self):
        """Si moe_enabled=true, ejecuta MoE."""
        registry = ExpertRegistry()
        baseline = MockExpert("baseline", _prediction=10.0)
        registry.register("baseline", baseline, baseline.capabilities)
        
        gateway = MoEGateway(
            registry=registry,
            gating=MockGatingNetwork({"baseline": 1.0}),
            fusion=SparseFusionLayer(),
            fallback_engine=MockFallbackEngine(),
            moe_enabled=True,
        )
        
        result = gateway.predict(create_test_window())
        
        assert "moe" in result.metadata
        assert result.metadata["moe"]["enabled"] is True


class TestSparsityK:
    """Tests de parámetro sparsity_k."""
    
    def test_sparsity_k_limits_experts(self):
        """sparsity_k limita número de expertos."""
        registry = ExpertRegistry()
        for name in ["e1", "e2", "e3"]:
            registry.register(name, MockExpert(name, _prediction=10.0), ExpertCapability())
        
        gateway = MoEGateway(
            registry=registry,
            gating=MockGatingNetwork({"e1": 0.5, "e2": 0.3, "e3": 0.2}),
            fusion=SparseFusionLayer(),
            fallback_engine=MockFallbackEngine(),
            sparsity_k=2,
        )
        
        result = gateway.predict(create_test_window())
        
        assert result.metadata["moe"]["sparsity_k"] == 2


class TestGatewayName:
    """Tests de interface PredictionPort."""
    
    def test_name_is_moe_gateway(self):
        """Nombre identificador del gateway."""
        gateway = MoEGateway(ExpertRegistry(), MockGatingNetwork({}), SparseFusionLayer(), MockFallbackEngine())
        assert gateway.name == "moe_gateway"
    
    def test_can_handle_delegates_when_disabled(self):
        """can_handle delega a fallback cuando MoE deshabilitado."""
        gateway = MoEGateway(
            ExpertRegistry(),
            MockGatingNetwork({}),
            SparseFusionLayer(),
            MockFallbackEngine(),
            moe_enabled=False,
        )
        
        assert gateway.can_handle(5) is True  # Fallback acepta >=1
