"""Tests for R-4 Seasonal Engine, MED-3 Memory Optimization, and Causal Narratives.

Validates:
1. FFT-based cycle detection in SeasonalPredictorEngine
2. __slots__ memory optimization in high-frequency classes
3. Causal narrative generation in explain phase
"""

import sys
import pytest
import numpy as np

# Direct imports to avoid circular dependency chain
sys.path.insert(0, '/home/nicolas/Documentos/Iot_System/iot_machine_learning')

from infrastructure.ml.engines.seasonal.engine import (
    SeasonalPredictorEngine,
    SeasonalConfig,
)
from infrastructure.ml.cognitive.orchestration.phases.explain_phase import (
    CausalNarrativeBuilder,
)
from domain.entities.iot.sensor_reading import (
    SensorReading, 
    SensorWindow,
)
from infrastructure.ml.interfaces import PredictionResult


class TestSeasonalPredictorEngine:
    """R-4: FFT-based seasonal prediction."""
    
    def test_detects_simple_cycle(self):
        """Engine should detect a simple sine wave cycle."""
        engine = SeasonalPredictorEngine()
        
        # Generate sine wave with period 10
        t = np.linspace(0, 4 * np.pi, 40)  # 2 full cycles
        values = list(np.sin(t))
        
        result = engine.predict(values)
        
        assert result.predicted_value is not None
        assert result.confidence > 0.3
        assert result.metadata.get("detected_period") is not None
    
    def test_handles_no_clear_cycle(self):
        """Should fallback when no clear seasonality."""
        engine = SeasonalPredictorEngine()
        
        # Random values (no cycle)
        np.random.seed(42)
        values = list(np.random.randn(20))
        
        result = engine.predict(values)
        
        # Should still return a result (fallback)
        assert result.predicted_value is not None
        assert result.metadata.get("fallback") is True
    
    def test_latency_under_10ms(self):
        """R-4: Must complete in <10ms for 50-point window."""
        import time
        engine = SeasonalPredictorEngine()
        
        # Generate 50-point window
        t = np.linspace(0, 4 * np.pi, 50)
        values = list(np.sin(t))
        
        start = time.perf_counter()
        result = engine.predict(values)
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        assert elapsed_ms < 10.0, f"Latency {elapsed_ms:.2f}ms exceeds 10ms budget"
    
    def test_can_handle_threshold(self):
        """Should require at least 2 cycles worth of data."""
        engine = SeasonalPredictorEngine(
            SeasonalConfig(min_period=5)
        )
        
        # Too few points (< 2 * min_period)
        assert not engine.can_handle(8)
        
        # Enough points
        assert engine.can_handle(10)
    
    def test_seasonal_trend_classification(self):
        """Should classify trend based on cycle phase."""
        engine = SeasonalPredictorEngine()
        
        # Rising sine wave (positive derivative)
        t = np.linspace(0, np.pi / 2, 10)  # First quarter (rising)
        values = list(np.sin(t))
        
        result = engine.predict(values)
        
        assert result.trend in ["up", "stable"]


class TestMemoryOptimization:
    """MED-3: __slots__ reduces memory overhead."""
    
    def test_sensor_reading_has_slots(self):
        """SensorReading should use __slots__."""
        # Check if the class has __slots__
        assert hasattr(SensorReading, '__slots__')
        
        # Verify no __dict__ (slots classes don't have dict)
        reading = SensorReading(
            sensor_id=1,
            value=25.0,
            timestamp=1234567890.0,
            sensor_type="temperature"
        )
        assert not hasattr(reading, '__dict__')
    
    def test_sensor_window_has_slots(self):
        """SensorWindow should use __slots__."""
        assert hasattr(SensorWindow, '__slots__')
        
        window = SensorWindow(sensor_id=1)
        assert not hasattr(window, '__dict__')
    
    def test_prediction_result_has_slots(self):
        """PredictionResult should use __slots__."""
        assert hasattr(PredictionResult, '__slots__')
        
        result = PredictionResult(
            predicted_value=25.0,
            confidence=0.85,
            trend="up",
        )
        assert not hasattr(result, '__dict__')


class MockProfile:
    """Mock signal profile for testing narratives."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockPerception:
    """Mock engine perception for testing narratives."""
    def __init__(self, engine_name, inhibited=False):
        self.engine_name = engine_name
        self.inhibited = inhibited


class TestCausalNarratives:
    """Causal narrative generation from metrics."""
    
    def test_anomaly_narrative(self):
        """z_score > 2.5 should generate anomaly narrative."""
        profile = MockProfile(z_score=3.0)
        
        narratives = CausalNarrativeBuilder.from_signal_profile(profile)
        
        assert any("anomalía" in n for n in narratives)
        assert any("cambio súbito" in n for n in narratives)
    
    def test_stability_narrative(self):
        """stability < 0.3 should generate stability narrative."""
        profile = MockProfile(stability=0.2)
        
        narratives = CausalNarrativeBuilder.from_signal_profile(profile)
        
        assert any("inestabilidad" in n for n in narratives)
        assert any("conservadora" in n for n in narratives)
    
    def test_volatile_regime_narrative(self):
        """VOLATILE regime should generate volatility narrative."""
        profile = MockProfile(regime="VOLATILE")
        
        narratives = CausalNarrativeBuilder.from_signal_profile(profile)
        
        assert any("volatilidad" in n for n in narratives)
    
    def test_trending_regime_narrative(self):
        """TRENDING regime should generate trend narrative."""
        profile = MockProfile(regime="TRENDING", trend_direction="up")
        
        narratives = CausalNarrativeBuilder.from_signal_profile(profile)
        
        assert any("Tendencia up" in n for n in narratives)
    
    def test_noise_narrative(self):
        """noise_ratio > 0.3 should generate noise narrative."""
        profile = MockProfile(noise_ratio=0.4)
        
        narratives = CausalNarrativeBuilder.from_signal_profile(profile)
        
        assert any("ruido" in n for n in narratives)
    
    def test_inhibition_majority_narrative(self):
        """Majority inhibited should generate fallback narrative."""
        perceptions = [
            MockPerception("taylor", inhibited=True),
            MockPerception("baseline", inhibited=True),
            MockPerception("seasonal", inhibited=False),
        ]
        
        narratives = CausalNarrativeBuilder.from_perceptions(perceptions)
        
        assert any("inhibidos" in n for n in narratives)
        assert any("fallback" in n for n in narratives)
    
    def test_single_active_engine_narrative(self):
        """Single active engine should be noted."""
        perceptions = [
            MockPerception("taylor", inhibited=False),
        ]
        
        narratives = CausalNarrativeBuilder.from_perceptions(perceptions)
        
        assert any("Engine único activo" in n for n in narratives)
    
    def test_empty_profile_no_narratives(self):
        """None profile should return empty list."""
        narratives = CausalNarrativeBuilder.from_signal_profile(None)
        assert narratives == []


class TestSeasonalEngineRegistration:
    """R-4: Engine registration in factory."""
    
    def test_seasonal_engine_registered(self):
        """Seasonal engine should be in factory registry."""
        from iot_machine_learning.infrastructure.ml.engines.core.factory import (
            EngineFactory,
        )
        
        engines = EngineFactory.list_engines()
        assert "seasonal_fft" in engines
    
    def test_can_create_seasonal_engine(self):
        """Factory should be able to create seasonal engine."""
        from iot_machine_learning.infrastructure.ml.engines.core.factory import (
            EngineFactory,
        )
        
        engine = EngineFactory.create("seasonal_fft")
        assert engine.name == "seasonal_fft"
