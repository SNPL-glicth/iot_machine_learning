"""Integration tests for all phases — 3A, 3B, 3C, and 4A.

Comprehensive integration tests for all cognitive phases.
"""

import pytest
from unittest.mock import Mock, MagicMock


class TestAllPhasesIntegration:
    """Integration tests for all cognitive phases."""
    
    def test_all_phase_components_exported(self):
        """Test all phase components are exported in cognitive/__init__.py."""
        from iot_machine_learning.infrastructure.ml.cognitive import (
            # Phase 3C - Observability
            CognitiveMetricsCollector,
            MemoryHealthMonitor,
            DriftDetectionEngine,
            ExplainabilityValidator,
            FeedbackLoopManager,
            CognitiveObservabilityDashboard,
            # Phase 3A - Memory
            SemanticEventBuilder,
            AnomalyMemoryStore,
            OperationalMemoryPipeline,
            HistoricalSimilarityRetriever,
            CognitiveMemoryRegistry,
            # Phase 3B - Explainability
            ContextualExplainabilityEngine,
            HistoricalContextAggregator,
            RecommendationGenerator,
            ContextualConfidenceCalculator,
            OperationalSummaryBuilder,
            # Phase 4A - Causal
            CausalCorrelationEngine,
            OperationalDependencyGraphManager,
            TemporalPatternMiner,
            EventPropagationTracker,
            PropagationConfidenceCalculator,
            OperationalSequenceRegistry,
        )
        
        # Verify all components are importable
        assert CognitiveMetricsCollector is not None
        assert MemoryHealthMonitor is not None
        assert DriftDetectionEngine is not None
        assert ExplainabilityValidator is not None
        assert FeedbackLoopManager is not None
        assert CognitiveObservabilityDashboard is not None
        assert SemanticEventBuilder is not None
        assert AnomalyMemoryStore is not None
        assert OperationalMemoryPipeline is not None
        assert HistoricalSimilarityRetriever is not None
        assert CognitiveMemoryRegistry is not None
        assert ContextualExplainabilityEngine is not None
        assert HistoricalContextAggregator is not None
        assert RecommendationGenerator is not None
        assert ContextualConfidenceCalculator is not None
        assert OperationalSummaryBuilder is not None
        assert CausalCorrelationEngine is not None
        assert OperationalDependencyGraphManager is not None
        assert TemporalPatternMiner is not None
        assert EventPropagationTracker is not None
        assert PropagationConfidenceCalculator is not None
        assert OperationalSequenceRegistry is not None
    
    def test_orchestrator_has_all_components(self):
        """Test MetaCognitiveOrchestrator has all phase components."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import MetaCognitiveOrchestrator
        from iot_machine_learning.infrastructure.ml.cognitive import (
            CognitiveMetricsCollector,
            MemoryHealthMonitor,
            CognitiveMemoryRegistry,
        )
        
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
        
        # Create orchestrator with all components
        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            metrics_collector=CognitiveMetricsCollector(),
            memory_health_monitor=MemoryHealthMonitor(),
            memory_registry=CognitiveMemoryRegistry(),
        )
        
        # Verify all components are set
        assert orchestrator._metrics_collector is not None
        assert orchestrator._memory_health_monitor is not None
        assert orchestrator._memory_registry is not None
    
    def test_pipeline_context_has_all_components(self):
        """Test PipelineContext includes all phase components."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import create_initial_context
        from iot_machine_learning.infrastructure.ml.cognitive import (
            CognitiveMetricsCollector,
            MemoryHealthMonitor,
            CognitiveMemoryRegistry,
        )
        
        # Create mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator._metrics_collector = CognitiveMetricsCollector()
        mock_orchestrator._memory_health_monitor = MemoryHealthMonitor()
        mock_orchestrator._memory_registry = CognitiveMemoryRegistry()
        
        # Create context
        ctx = create_initial_context(
            orchestrator=mock_orchestrator,
            values=[1.0, 2.0, 3.0],
            timestamps=[1000.0, 1001.0, 1002.0],
            series_id="test_series",
            flags={},
            timer=Mock(total_ms=100.0, budget_ms=500.0),
        )
        
        # Verify all components are in context
        assert ctx.metrics_collector is not None
        assert ctx.memory_health_monitor is not None
        assert ctx.memory_registry is not None
    
    def test_all_phases_in_pipeline_executor(self):
        """Test all new phases are included in default pipeline."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor import PipelineExecutor
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.memory_phase import MemoryPhase
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.observability_phase import ObservabilityPhase
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.causal_phase import CausalPhase
        
        # Create pipeline executor with default phases
        executor = PipelineExecutor()
        
        # Verify all new phases are in the pipeline
        phase_names = [phase.name for phase in executor._phases]
        assert "memory" in phase_names
        assert "observability" in phase_names
        assert "causal" in phase_names
    
    def test_phase_3c_integration_complete(self):
        """Test Phase 3C integration is complete."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.drift_detection_phase import DriftDetectionPhase
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.explain_phase import ExplainPhase
        from iot_machine_learning.infrastructure.ml.cognitive import DriftDetectionEngine, ExplainabilityValidator
        
        # Verify DriftDetectionPhase can use DriftDetectionEngine
        drift_phase = DriftDetectionPhase(drift_detection_engine=DriftDetectionEngine())
        assert drift_phase._drift_detection_engine is not None
        
        # Verify ExplainPhase can use ExplainabilityValidator
        explain_phase = ExplainPhase(explainability_validator=ExplainabilityValidator())
        assert explain_phase._explainability_validator is not None
    
    def test_phase_3a_integration_complete(self):
        """Test Phase 3A integration is complete."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.memory_phase import MemoryPhase
        from iot_machine_learning.infrastructure.ml.cognitive import SemanticEventBuilder, AnomalyMemoryStore
        
        # Verify MemoryPhase can use memory components
        memory_phase = MemoryPhase(
            semantic_event_builder=SemanticEventBuilder(),
            anomaly_memory_store=AnomalyMemoryStore(),
        )
        assert memory_phase._semantic_event_builder is not None
        assert memory_phase._anomaly_memory_store is not None
    
    def test_phase_3b_integration_complete(self):
        """Test Phase 3B integration is complete."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.explain_phase import ExplainPhase
        from iot_machine_learning.infrastructure.ml.cognitive import ContextualExplainabilityEngine
        
        # Verify ExplainPhase can use ContextualExplainabilityEngine
        explain_phase = ExplainPhase(contextual_explainability_engine=ContextualExplainabilityEngine())
        assert explain_phase._contextual_explainability_engine is not None
    
    def test_phase_4a_integration_complete(self):
        """Test Phase 4A integration is complete."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.causal_phase import CausalPhase
        from iot_machine_learning.infrastructure.ml.cognitive import (
            CausalCorrelationEngine,
            EventPropagationTracker,
            PropagationConfidenceCalculator,
        )
        
        # Verify CausalPhase can use causal components
        causal_phase = CausalPhase(
            causal_correlation_engine=CausalCorrelationEngine(),
            event_propagation_tracker=EventPropagationTracker(),
            propagation_confidence_calculator=PropagationConfidenceCalculator(),
        )
        assert causal_phase._causal_correlation_engine is not None
        assert causal_phase._event_propagation_tracker is not None
        assert causal_phase._propagation_confidence_calculator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
