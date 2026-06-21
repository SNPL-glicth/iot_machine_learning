"""Integration tests for Phase 3C — Cognitive Hardening & Observability.

Tests that all observability components are properly integrated into the pipeline.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time


class TestPhase3CIntegration:
    """Integration tests for Phase 3C observability components."""
    
    def test_cognitive_metrics_collector_integration(self):
        """Test CognitiveMetricsCollector is integrated in orchestrator."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import MetaCognitiveOrchestrator
        from iot_machine_learning.infrastructure.ml.cognitive import CognitiveMetricsCollector
        
        # Create mock engines
        mock_engine = Mock()
        mock_engine.name = "test_engine"
        mock_engine.can_handle.return_value = True
        mock_engine.predict.return_value = Mock(
            predicted_value=10.0,
            confidence=0.8,
            trend="stable",
            metadata={}
        )
        
        # Create orchestrator with metrics collector
        metrics_collector = CognitiveMetricsCollector()
        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            metrics_collector=metrics_collector,
        )
        
        # Verify metrics collector is set
        assert orchestrator._metrics_collector is not None
        assert isinstance(orchestrator._metrics_collector, CognitiveMetricsCollector)
    
    def test_memory_health_monitor_integration(self):
        """Test MemoryHealthMonitor is integrated in orchestrator."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import MetaCognitiveOrchestrator
        from iot_machine_learning.infrastructure.ml.cognitive import MemoryHealthMonitor
        
        # Create mock engines
        mock_engine = Mock()
        mock_engine.name = "test_engine"
        mock_engine.can_handle.return_value = True
        mock_engine.predict.return_value = Mock(
            predicted_value=10.0,
            confidence=0.8,
            trend="stable",
            metadata={}
        )
        
        # Create orchestrator with memory health monitor
        memory_health_monitor = MemoryHealthMonitor()
        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            memory_health_monitor=memory_health_monitor,
        )
        
        # Verify memory health monitor is set
        assert orchestrator._memory_health_monitor is not None
        assert isinstance(orchestrator._memory_health_monitor, MemoryHealthMonitor)
    
    def test_pipeline_context_has_observability_components(self):
        """Test PipelineContext includes observability components."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import create_initial_context
        from iot_machine_learning.infrastructure.ml.cognitive import CognitiveMetricsCollector, MemoryHealthMonitor
        
        # Create mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator._metrics_collector = CognitiveMetricsCollector()
        mock_orchestrator._memory_health_monitor = MemoryHealthMonitor()
        
        # Create context
        ctx = create_initial_context(
            orchestrator=mock_orchestrator,
            values=[1.0, 2.0, 3.0],
            timestamps=[1000.0, 1001.0, 1002.0],
            series_id="test_series",
            flags={},
            timer=Mock(total_ms=100.0, budget_ms=500.0),
        )
        
        # Verify observability components are in context
        assert ctx.metrics_collector is not None
        assert ctx.memory_health_monitor is not None
    
    def test_drift_detection_engine_integration(self):
        """Test DriftDetectionEngine is integrated in DriftDetectionPhase."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.drift_detection_phase import DriftDetectionPhase
        from iot_machine_learning.infrastructure.ml.cognitive import DriftDetectionEngine
        
        # Create phase with drift detection engine
        drift_detection_engine = DriftDetectionEngine()
        phase = DriftDetectionPhase(drift_detection_engine=drift_detection_engine)
        
        # Verify drift detection engine is set
        assert phase._drift_detection_engine is not None
        assert isinstance(phase._drift_detection_engine, DriftDetectionEngine)
    
    def test_explainability_validator_integration(self):
        """Test ExplainabilityValidator is integrated in ExplainPhase."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.explain_phase import ExplainPhase
        from iot_machine_learning.infrastructure.ml.cognitive import ExplainabilityValidator
        
        # Create phase with explainability validator
        explainability_validator = ExplainabilityValidator()
        phase = ExplainPhase(explainability_validator=explainability_validator)
        
        # Verify explainability validator is set
        assert phase._explainability_validator is not None
        assert isinstance(phase._explainability_validator, ExplainabilityValidator)
    
    def test_observability_phase_exists(self):
        """Test ObservabilityPhase is created and can be instantiated."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.observability_phase import ObservabilityPhase
        
        # Create phase
        phase = ObservabilityPhase()
        
        # Verify phase exists and has correct name
        assert phase.name == "observability"
    
    def test_feedback_loop_manager_integration(self):
        """Test FeedbackLoopManager is integrated in AdaptPhase."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.adapt_phase import AdaptPhase
        from iot_machine_learning.infrastructure.ml.cognitive import FeedbackLoopManager
        
        # Create phase with feedback loop manager
        feedback_loop_manager = FeedbackLoopManager()
        phase = AdaptPhase(feedback_loop_manager=feedback_loop_manager)
        
        # Verify feedback loop manager is set
        assert phase._feedback_loop_manager is not None
        assert isinstance(phase._feedback_loop_manager, FeedbackLoopManager)
    
    def test_observability_phase_in_pipeline_executor(self):
        """Test ObservabilityPhase is included in default pipeline."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor import PipelineExecutor
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.observability_phase import ObservabilityPhase
        
        # Create pipeline executor with default phases
        executor = PipelineExecutor()
        
        # Verify ObservabilityPhase is in the pipeline
        phase_names = [phase.name for phase in executor._phases]
        assert "observability" in phase_names
    
    def test_phase_3c_components_exported(self):
        """Test Phase 3C components are exported in cognitive/__init__.py."""
        from iot_machine_learning.infrastructure.ml.cognitive import (
            CognitiveMetricsCollector,
            MemoryHealthMonitor,
            DriftDetectionEngine,
            ExplainabilityValidator,
            FeedbackLoopManager,
            CognitiveObservabilityDashboard,
        )
        
        # Verify all components are importable
        assert CognitiveMetricsCollector is not None
        assert MemoryHealthMonitor is not None
        assert DriftDetectionEngine is not None
        assert ExplainabilityValidator is not None
        assert FeedbackLoopManager is not None
        assert CognitiveObservabilityDashboard is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
