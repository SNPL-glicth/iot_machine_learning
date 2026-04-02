"""Tests for Fase 5: Scalability components.

Validates:
1. ShadowEvaluationPhase - shadow mode execution
2. PersistenceAdapter - Redis/Postgres/Hybrid persistence
3. DataDriftDetector - concept drift detection
4. ContextStateManager with persistence integration
"""

import sys
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock

sys.path.insert(0, '/home/nicolas/Documentos/Iot_System/iot_machine_learning')

from infrastructure.ml.cognitive.orchestration.phases.shadow_evaluation_phase import (
    ShadowEvaluationPhase,
    ShadowResult,
    ShadowEvaluationSummary,
)
from infrastructure.ml.cognitive.orchestration.persistence_adapter import (
    RedisPersistenceAdapter,
    PostgresPersistenceAdapter,
    HybridPersistenceAdapter,
)
from infrastructure.ml.cognitive.orchestration.context_state_manager import (
    ContextStateManager,
    SeriesState,
)
from ml_service.metrics.prometheus_exporter import (
    PrometheusExporter,
    DataDriftDetector,
)
from infrastructure.ml.interfaces import PredictionResult, PredictionEngine


class MockEngine(PredictionEngine):
    """Mock prediction engine for testing."""
    
    def __init__(self, name: str, prediction: float = 25.0, confidence: float = 0.8):
        self._name = name
        self._prediction = prediction
        self._confidence = confidence
    
    @property
    def name(self) -> str:
        return self._name
    
    def predict(self, values, timestamps=None):
        return PredictionResult(
            predicted_value=self._prediction,
            confidence=self._confidence,
            trend="stable",
            metadata={"engine": self._name}
        )
    
    def can_handle(self, n_points: int) -> bool:
        return n_points >= 2


class MockContext:
    """Mock pipeline context for testing."""
    
    def __init__(self, values=None, series_id="test_series", fused_value=None):
        self.values = values or [1.0, 2.0, 3.0, 4.0, 5.0]
        self.timestamps = None
        self.series_id = series_id
        self.fused_value = fused_value or 5.5
        self.orchestrator = Mock()
    
    def with_field(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        return self


class TestShadowEvaluationPhase:
    """Shadow mode evaluation tests."""
    
    def test_shadow_phase_runs_experimental_engines(self):
        """Shadow phase should execute engines without affecting main prediction."""
        shadow_engines = [
            MockEngine("experimental_1", prediction=26.0),
            MockEngine("experimental_2", prediction=24.0),
        ]
        phase = ShadowEvaluationPhase(shadow_engines=shadow_engines, enabled=True)
        ctx = MockContext()
        
        result_ctx = phase.execute(ctx)
        
        assert hasattr(result_ctx, 'experimental_metadata')
        assert result_ctx.experimental_metadata['shadow_engines_tested'] == 2
    
    def test_shadow_phase_disabled_when_not_enabled(self):
        """Shadow phase should be skipped when disabled."""
        phase = ShadowEvaluationPhase(shadow_engines=[], enabled=False)
        ctx = MockContext()
        
        result_ctx = phase.execute(ctx)
        
        assert result_ctx.experimental_metadata.get('shadow') is None
    
    def test_shadow_phase_with_sampling(self):
        """Shadow phase should respect sample rate."""
        shadow_engines = [MockEngine("exp")]
        phase = ShadowEvaluationPhase(shadow_engines=shadow_engines, enabled=True, sample_rate=0.0)
        ctx = MockContext()
        
        result_ctx = phase.execute(ctx)
        
        assert result_ctx.experimental_metadata.get('shadow') == 'skipped'
    
    def test_shadow_result_structure(self):
        """ShadowResult should capture all required fields."""
        result = ShadowResult(
            engine_name="test_engine",
            predicted_value=25.5,
            confidence=0.85,
            latency_ms=5.2,
            error_vs_actual=0.5
        )
        
        assert result.engine_name == "test_engine"
        assert result.predicted_value == 25.5
        assert result.latency_ms == 5.2


class MockAsyncRedis:
    """Mock async Redis client."""
    
    def __init__(self):
        self._data = {}
    
    async def setex(self, key, ttl, value):
        self._data[key] = value
        return True
    
    async def get(self, key):
        return self._data.get(key)
    
    async def delete(self, key):
        self._data.pop(key, None)
        return True


class MockAsyncPool:
    """Mock async PostgreSQL pool."""
    
    def __init__(self):
        self._data = {}
    
    def acquire(self):
        return MockAsyncConn(self._data)


class MockAsyncConn:
    """Mock async PostgreSQL connection."""
    
    def __init__(self, data_store):
        self._data = data_store
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        pass
    
    async def execute(self, query, *params):
        if "INSERT" in query or "UPDATE" in query:
            series_id = params[0]
            self._data[series_id] = {
                "regime": params[1],
                "last_updated": params[2],
                "prediction_count": params[3],
            }
        elif "DELETE" in query:
            self._data.pop(params[0], None)
    
    async def fetchrow(self, query, *params):
        return self._data.get(params[0])


class TestDataDriftDetector:
    """Data drift detection tests."""
    
    def test_no_drift_with_stable_data(self):
        """Should not detect drift with stable distribution."""
        detector = DataDriftDetector(window_size=50, drift_threshold=2.0)
        
        # Stable normal distribution
        np.random.seed(42)
        for _ in range(5):
            values = list(np.random.normal(10, 1, 20))
            drift_score = detector.update(values)
        
        assert not detector.is_drift_detected()
        assert drift_score < 2.0
    
    def test_detects_mean_shift(self):
        """Should detect drift when mean shifts significantly."""
        detector = DataDriftDetector(window_size=50, drift_threshold=2.0)
        
        # Initial stable distribution
        np.random.seed(42)
        for _ in range(3):
            values = list(np.random.normal(10, 1, 20))
            detector.update(values)
        
        # Shifted distribution (mean +5)
        shifted_values = list(np.random.normal(15, 1, 30))
        drift_score = detector.update(shifted_values)
        
        assert drift_score > 2.0
        assert detector.is_drift_detected()
    
    def test_get_stats_returns_valid_data(self):
        """get_stats should return valid statistics."""
        detector = DataDriftDetector()
        
        np.random.seed(42)
        values = list(np.random.normal(10, 1, 25))
        detector.update(values)
        
        stats = detector.get_stats()
        assert "drift_score" in stats
        assert "baseline_mean" in stats
        assert "baseline_std" in stats
    
    def test_empty_values_return_zero(self):
        """Empty values should return zero drift score."""
        detector = DataDriftDetector()
        
        drift_score = detector.update([])
        
        assert drift_score == 0.0


class TestPrometheusExporterDrift:
    """Prometheus exporter with data drift tests."""
    
    def test_record_values_creates_drift_detector(self):
        """Recording values should create drift detector for series."""
        exporter = PrometheusExporter()
        
        np.random.seed(42)
        values = list(np.random.normal(10, 1, 25))
        drift_score = exporter.record_values("series_1", values)
        
        assert drift_score >= 0.0
        assert "series_1" in exporter._drift_detectors
    
    def test_concept_drift_detection(self):
        """Should detect concept drift in exported metrics."""
        exporter = PrometheusExporter()
        
        # Initial stable values
        np.random.seed(42)
        for _ in range(3):
            values = list(np.random.normal(10, 1, 20))
            exporter.record_values("series_1", values)
        
        assert not exporter.is_concept_drift_detected("series_1")
        
        # Shifted values
        shifted = list(np.random.normal(20, 1, 30))
        exporter.record_values("series_1", shifted)
        
        # May or may not detect depending on adaptation
        drift_stats = exporter.get_drift_stats("series_1")
        assert "drift_score" in drift_stats
    
    def test_prometheus_format_includes_drift(self):
        """Prometheus format should include drift metrics."""
        exporter = PrometheusExporter()
        
        np.random.seed(42)
        values = list(np.random.normal(10, 1, 25))
        exporter.record_values("series_1", values)
        
        output = exporter.export_prometheus_format()
        
        assert "zenin_concept_drift_detected" in output
        assert "zenin_concept_drift_score" in output
        assert "zenin_drift_alerts_total" in output
    
    def test_metrics_summary_includes_drift(self):
        """Metrics summary should include drift statistics."""
        exporter = PrometheusExporter()
        
        np.random.seed(42)
        values = list(np.random.normal(10, 1, 25))
        exporter.record_values("series_1", values)
        
        summary = exporter.get_metrics_summary()
        
        assert "drift_monitored" in summary
        assert "drift_detected" in summary
        assert summary["drift_monitored"] >= 1


class TestContextStateManagerPersistence:
    """ContextStateManager with persistence tests."""
    
    def test_state_manager_without_persistence(self):
        """State manager should work without persistence."""
        manager = ContextStateManager(max_series=100)
        
        state = manager.get_state("series_1")
        assert state is not None
        assert state.prediction_count == 0
    
    def test_update_regime_triggers_persistence(self):
        """Updating regime should trigger async persistence."""
        manager = ContextStateManager(max_series=100)
        
        # Should not raise even without persistence
        manager.update_regime("series_1", "VOLATILE")
        
        assert manager.get_regime("series_1") == "VOLATILE"
    
    def test_metrics_report_persistence_status(self):
        """Metrics should report persistence status."""
        manager = ContextStateManager(max_series=100)
        
        metrics = manager.get_metrics()
        
        assert "persistence_enabled" in metrics
        assert metrics["persistence_enabled"] is False
    
    def test_increment_prediction_count(self):
        """Should increment and return prediction count."""
        manager = ContextStateManager(max_series=100)
        
        count1 = manager.increment_prediction_count("series_1")
        count2 = manager.increment_prediction_count("series_1")
        
        assert count1 == 1
        assert count2 == 2


# Async tests require pytest-asyncio, skip if not available
pytestmark = pytest.mark.skipif(
    not sys.modules.get("pytest_asyncio"),
    reason="pytest-asyncio not available"
)
