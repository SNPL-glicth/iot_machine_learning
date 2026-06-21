"""Integration tests for Neural and Decision components.

Tests that HybridNeuralEngine and ContextualDecisionEngine are properly integrated.
"""

import pytest
from unittest.mock import Mock, MagicMock


class TestNeuralDecisionIntegration:
    """Integration tests for neural and decision components."""
    
    def test_neural_components_exported(self):
        """Test Neural components are exported in cognitive/__init__.py."""
        from iot_machine_learning.infrastructure.ml.cognitive import HybridNeuralEngine, NeuralResult
        
        # Verify components are importable
        assert HybridNeuralEngine is not None
        assert NeuralResult is not None
    
    def test_decision_components_exported(self):
        """Test Decision components are exported in cognitive/__init__.py."""
        from iot_machine_learning.infrastructure.ml.cognitive import ContextualDecisionEngine
        
        # Verify component is importable
        assert ContextualDecisionEngine is not None
    
    def test_neural_engine_integration(self):
        """Test HybridNeuralEngine is integrated in orchestrator."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import MetaCognitiveOrchestrator
        from iot_machine_learning.infrastructure.ml.cognitive import HybridNeuralEngine
        
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
        
        # Create orchestrator with neural engine
        neural_engine = HybridNeuralEngine()
        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            neural_engine=neural_engine,
        )
        
        # Verify neural engine is set
        assert orchestrator._neural_engine is not None
        assert isinstance(orchestrator._neural_engine, HybridNeuralEngine)
    
    def test_decision_engine_integration(self):
        """Test ContextualDecisionEngine is integrated in orchestrator."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import MetaCognitiveOrchestrator
        from iot_machine_learning.infrastructure.ml.cognitive import ContextualDecisionEngine
        
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
        
        # Create orchestrator with decision engine
        decision_engine = ContextualDecisionEngine()
        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            decision_engine=decision_engine,
        )
        
        # Verify decision engine is set
        assert orchestrator._decision_engine is not None
        assert isinstance(orchestrator._decision_engine, ContextualDecisionEngine)
    
    def test_pipeline_context_has_neural_and_decision(self):
        """Test PipelineContext includes neural and decision engines."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import create_initial_context
        from iot_machine_learning.infrastructure.ml.cognitive import HybridNeuralEngine, ContextualDecisionEngine
        
        # Create mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator._metrics_collector = None
        mock_orchestrator._memory_health_monitor = None
        mock_orchestrator._memory_registry = None
        mock_orchestrator._neural_engine = HybridNeuralEngine()
        mock_orchestrator._decision_engine = ContextualDecisionEngine()
        
        # Create context
        ctx = create_initial_context(
            orchestrator=mock_orchestrator,
            values=[1.0, 2.0, 3.0],
            timestamps=[1000.0, 1001.0, 1002.0],
            series_id="test_series",
            flags={},
            timer=Mock(total_ms=100.0, budget_ms=500.0),
        )
        
        # Verify neural and decision engines are in context
        assert ctx.neural_engine is not None
        assert ctx.decision_engine is not None
    
    def test_neural_phase_exists(self):
        """Test NeuralPhase is created and can be instantiated."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.neural_phase import NeuralPhase
        
        # Create phase
        phase = NeuralPhase()
        
        # Verify phase exists and has correct name
        assert phase.name == "neural"
    
    def test_decision_phase_exists(self):
        """Test DecisionPhase is created and can be instantiated."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.decision_phase import DecisionPhase
        
        # Create phase
        phase = DecisionPhase()
        
        # Verify phase exists and has correct name
        assert phase.name == "decision"
    
    def test_neural_phase_in_pipeline_executor(self):
        """Test NeuralPhase is included in default pipeline."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor import PipelineExecutor
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.neural_phase import NeuralPhase
        
        # Create pipeline executor with default phases
        executor = PipelineExecutor()
        
        # Verify NeuralPhase is in the pipeline
        phase_names = [phase.name for phase in executor._phases]
        assert "neural" in phase_names
    
    def test_decision_phase_in_pipeline_executor(self):
        """Test DecisionPhase is included in default pipeline."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor import PipelineExecutor
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.decision_phase import DecisionPhase
        
        # Create pipeline executor with default phases
        executor = PipelineExecutor()
        
        # Verify DecisionPhase is in the pipeline
        phase_names = [phase.name for phase in executor._phases]
        assert "decision" in phase_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
