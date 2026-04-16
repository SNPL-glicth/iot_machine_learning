"""Tests for expert adapters (Baseline, Statistical, Taylor).

Verifies Adapter pattern: wraps engines without modification.
"""

import pytest
from dataclasses import dataclass
from typing import List, Optional

from iot_machine_learning.domain.ports.expert_port import ExpertOutput
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.infrastructure.ml.interfaces import PredictionEngine, PredictionResult

from ..baseline_expert import BaselineExpert
from ..statistical_expert import StatisticalExpert
from ..taylor_expert import TaylorExpert
from ...registry.expert_capability import ExpertCapability


@dataclass
class MockPredictionEngine(PredictionEngine):
    """Mock engine for testing adapters."""
    
    _name: str
    _predicted_value: float = 10.0
    _confidence: float = 0.8
    _trend: str = "stable"
    _min_points: int = 3
    
    @property
    def name(self) -> str:
        return self._name
    
    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None
    ) -> PredictionResult:
        return PredictionResult(
            predicted_value=self._predicted_value,
            confidence=self._confidence,
            trend=self._trend,
            metadata={"test": True},
        )
    
    def can_handle(self, n_points: int) -> bool:
        return n_points >= self._min_points


def create_test_window(n_points: int = 5) -> SensorWindow:
    """Helper to create test window."""
    from iot_machine_learning.domain.entities.iot.sensor_reading import Reading
    readings = [
        Reading(series_id="42", value=float(20 + i), timestamp=1000.0 + i)
        for i in range(n_points)
    ]
    return SensorWindow(series_id="42", readings=readings)


class TestBaselineExpert:
    """Tests for BaselineExpert adapter."""
    
    def test_adapter_name(self):
        """Expert name is 'baseline'."""
        engine = MockPredictionEngine("baseline_engine")
        expert = BaselineExpert(engine)
        
        assert expert.name == "baseline"
    
    def test_capabilities(self):
        """Capabilities configured for stable regime."""
        engine = MockPredictionEngine("baseline_engine")
        expert = BaselineExpert(engine)
        
        caps = expert.capabilities
        
        assert "stable" in caps.regimes
        assert caps.computational_cost == 1.0
        assert caps.min_points == 3
    
    def test_predict_delegates_to_engine(self):
        """Predict delegates to wrapped engine."""
        engine = MockPredictionEngine(
            "baseline_engine",
            _predicted_value=25.0,
            _confidence=0.9,
        )
        expert = BaselineExpert(engine)
        window = create_test_window(5)
        
        result = expert.predict(window)
        
        assert isinstance(result, ExpertOutput)
        assert result.prediction == 25.0
        assert result.confidence == 0.9
    
    def test_can_handle_delegates_to_engine(self):
        """Can_handle delegates to wrapped engine."""
        engine = MockPredictionEngine("baseline_engine", _min_points=5)
        expert = BaselineExpert(engine)
        
        sufficient_window = create_test_window(5)
        insufficient_window = create_test_window(3)
        
        assert expert.can_handle(sufficient_window) is True
        assert expert.can_handle(insufficient_window) is False


class TestStatisticalExpert:
    """Tests for StatisticalExpert adapter."""
    
    def test_adapter_name(self):
        """Expert name is 'statistical'."""
        engine = MockPredictionEngine("statistical_engine")
        expert = StatisticalExpert(engine)
        
        assert expert.name == "statistical"
    
    def test_capabilities(self):
        """Capabilities configured for stable and trending."""
        engine = MockPredictionEngine("statistical_engine")
        expert = StatisticalExpert(engine)
        
        caps = expert.capabilities
        
        assert "stable" in caps.regimes
        assert "trending" in caps.regimes
        assert caps.computational_cost == 1.5
        assert caps.min_points == 5
        assert "seasonality" in caps.specialties
    
    def test_predict_delegates_to_engine(self):
        """Predict delegates to wrapped engine."""
        engine = MockPredictionEngine(
            "statistical_engine",
            _predicted_value=30.0,
            _confidence=0.85,
            _trend="up",
        )
        expert = StatisticalExpert(engine)
        window = create_test_window(5)
        
        result = expert.predict(window)
        
        assert result.prediction == 30.0
        assert result.confidence == 0.85
        assert result.trend == "up"
    
    def test_metadata_includes_method(self):
        """Metadata identifies statistical method."""
        engine = MockPredictionEngine("statistical_engine")
        expert = StatisticalExpert(engine)
        window = create_test_window(5)
        
        result = expert.predict(window)
        
        assert result.metadata.get("method") == "ema_holt"


class TestTaylorExpert:
    """Tests for TaylorExpert adapter."""
    
    def test_adapter_name(self):
        """Expert name is 'taylor'."""
        engine = MockPredictionEngine("taylor_engine")
        expert = TaylorExpert(engine)
        
        assert expert.name == "taylor"
    
    def test_capabilities(self):
        """Capabilities configured for volatile and trending."""
        engine = MockPredictionEngine("taylor_engine")
        expert = TaylorExpert(engine)
        
        caps = expert.capabilities
        
        assert "volatile" in caps.regimes
        assert "trending" in caps.regimes
        assert caps.computational_cost == 2.5
        assert caps.min_points == 5
        assert "non_linear" in caps.specialties
        assert "derivatives" in caps.specialties
    
    def test_predict_delegates_to_engine(self):
        """Predict delegates to wrapped engine."""
        engine = MockPredictionEngine(
            "taylor_engine",
            _predicted_value=35.0,
            _confidence=0.75,
            _trend="down",
        )
        expert = TaylorExpert(engine)
        window = create_test_window(5)
        
        result = expert.predict(window)
        
        assert result.prediction == 35.0
        assert result.confidence == 0.75
        assert result.trend == "down"
    
    def test_metadata_includes_method(self):
        """Metadata identifies Taylor method."""
        engine = MockPredictionEngine("taylor_engine")
        expert = TaylorExpert(engine)
        window = create_test_window(5)
        
        result = expert.predict(window)
        
        assert result.metadata.get("method") == "taylor_series"


class TestAdapterPattern:
    """Tests verifying Adapter pattern compliance."""
    
    def test_does_not_modify_engine(self):
        """Adapter does not modify wrapped engine."""
        engine = MockPredictionEngine("original")
        original_name = engine.name
        
        expert = BaselineExpert(engine)
        
        # Engine unchanged
        assert engine.name == original_name
        assert engine._name == "original"
    
    def test_multiple_adapters_same_engine_type(self):
        """Can create multiple adapters for same engine type."""
        engine1 = MockPredictionEngine("engine_1")
        engine2 = MockPredictionEngine("engine_2")
        
        expert1 = BaselineExpert(engine1)
        expert2 = BaselineExpert(engine2)
        
        assert expert1.name == "baseline"
        assert expert2.name == "baseline"
        # Both are independent
        assert expert1._engine is engine1
        assert expert2._engine is engine2
    
    def test_estimate_latency_scales_with_n_points(self):
        """Latency estimate increases with n_points."""
        engine = MockPredictionEngine("baseline_engine")
        expert = BaselineExpert(engine)
        
        latency_10 = expert.estimate_latency_ms(10)
        latency_100 = expert.estimate_latency_ms(100)
        
        assert latency_100 > latency_10


class TestExpertCapabilitiesComparison:
    """Compare capabilities across expert types."""
    
    def test_computational_cost_order(self):
        """Costs reflect computational complexity."""
        baseline = BaselineExpert(MockPredictionEngine("b"))
        statistical = StatisticalExpert(MockPredictionEngine("s"))
        taylor = TaylorExpert(MockPredictionEngine("t"))
        
        # Baseline < Statistical < Taylor
        assert baseline.capabilities.computational_cost < statistical.capabilities.computational_cost
        assert statistical.capabilities.computational_cost < taylor.capabilities.computational_cost
    
    def test_regime_coverage(self):
        """Different experts cover different regimes."""
        baseline = BaselineExpert(MockPredictionEngine("b"))
        statistical = StatisticalExpert(MockPredictionEngine("s"))
        taylor = TaylorExpert(MockPredictionEngine("t"))
        
        # Baseline: stable only
        assert baseline.capabilities.regimes == ("stable",)
        
        # Statistical: stable + trending
        assert "stable" in statistical.capabilities.regimes
        assert "trending" in statistical.capabilities.regimes
        
        # Taylor: volatile + trending
        assert "volatile" in taylor.capabilities.regimes
        assert "trending" in taylor.capabilities.regimes
