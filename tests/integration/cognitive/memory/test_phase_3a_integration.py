"""Integration tests for Phase 3A — Memory Foundation MVP.

Tests that all memory components are properly integrated into the pipeline.
"""

import pytest
from unittest.mock import Mock, MagicMock


class TestPhase3AIntegration:
    """Integration tests for Phase 3A memory components."""
    
    def test_memory_components_exported(self):
        """Test Phase 3A components are exported in cognitive/__init__.py."""
        from iot_machine_learning.infrastructure.ml.cognitive import (
            SemanticEventBuilder,
            AnomalyMemoryStore,
            OperationalMemoryPipeline,
            HistoricalSimilarityRetriever,
            CognitiveMemoryRegistry,
        )
        
        # Verify all components are importable
        assert SemanticEventBuilder is not None
        assert AnomalyMemoryStore is not None
        assert OperationalMemoryPipeline is not None
        assert HistoricalSimilarityRetriever is not None
        assert CognitiveMemoryRegistry is not None
    
    def test_cognitive_memory_registry_integration(self):
        """Test CognitiveMemoryRegistry is integrated in orchestrator."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import MetaCognitiveOrchestrator
        from iot_machine_learning.infrastructure.ml.cognitive import CognitiveMemoryRegistry
        
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
        
        # Create orchestrator with memory registry
        memory_registry = CognitiveMemoryRegistry()
        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            memory_registry=memory_registry,
        )
        
        # Verify memory registry is set
        assert orchestrator._memory_registry is not None
        assert isinstance(orchestrator._memory_registry, CognitiveMemoryRegistry)
    
    def test_pipeline_context_has_memory_registry(self):
        """Test PipelineContext includes memory registry."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import create_initial_context
        from iot_machine_learning.infrastructure.ml.cognitive import CognitiveMemoryRegistry
        
        # Create mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator._metrics_collector = None
        mock_orchestrator._memory_health_monitor = None
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
        
        # Verify memory registry is in context
        assert ctx.memory_registry is not None
    
    def test_memory_phase_exists(self):
        """Test MemoryPhase is created and can be instantiated."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.memory_phase import MemoryPhase
        
        # Create phase
        phase = MemoryPhase()
        
        # Verify phase exists and has correct name
        assert phase.name == "memory"
    
    def test_memory_phase_in_pipeline_executor(self):
        """Test MemoryPhase is included in default pipeline."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor import PipelineExecutor
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.memory_phase import MemoryPhase
        
        # Create pipeline executor with default phases
        executor = PipelineExecutor()
        
        # Verify MemoryPhase is in the pipeline
        phase_names = [phase.name for phase in executor._phases]
        assert "memory" in phase_names
    
    def test_semantic_event_builder_integration(self):
        """Test SemanticEventBuilder can be integrated in MemoryPhase."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.memory_phase import MemoryPhase
        from iot_machine_learning.infrastructure.ml.cognitive import SemanticEventBuilder
        
        # Create phase with semantic event builder
        semantic_event_builder = SemanticEventBuilder()
        phase = MemoryPhase(semantic_event_builder=semantic_event_builder)
        
        # Verify semantic event builder is set
        assert phase._semantic_event_builder is not None
        assert isinstance(phase._semantic_event_builder, SemanticEventBuilder)
    
    def test_anomaly_memory_store_integration(self):
        """Test AnomalyMemoryStore can be integrated in MemoryPhase."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.memory_phase import MemoryPhase
        from iot_machine_learning.infrastructure.ml.cognitive import AnomalyMemoryStore
        
        # Create phase with anomaly memory store
        anomaly_memory_store = AnomalyMemoryStore()
        phase = MemoryPhase(anomaly_memory_store=anomaly_memory_store)
        
        # Verify anomaly memory store is set
        assert phase._anomaly_memory_store is not None
        assert isinstance(phase._anomaly_memory_store, AnomalyMemoryStore)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
