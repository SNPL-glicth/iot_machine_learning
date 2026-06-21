"""Integration tests for Pattern, Dynamic, and Regime components.

Tests that PatternInterpreter, RollingWindowEngine, and RegimeDetectionPipeline are properly integrated.
"""

import pytest
from unittest.mock import Mock, MagicMock


class TestPatternDynamicRegimeIntegration:
    """Integration tests for pattern, dynamic, and regime components."""
    
    def test_pattern_components_exported(self):
        """Test Pattern components are exported in cognitive/__init__.py."""
        from iot_machine_learning.infrastructure.ml.cognitive import PatternInterpreter, InterpretedPattern
        
        # Verify components are importable
        assert PatternInterpreter is not None
        assert InterpretedPattern is not None
    
    def test_dynamic_components_exported(self):
        """Test Dynamic components are exported in cognitive/__init__.py."""
        from iot_machine_learning.infrastructure.ml.cognitive import RollingWindowEngine, DynamicFeaturePipeline
        
        # Verify components are importable
        assert RollingWindowEngine is not None
        assert DynamicFeaturePipeline is not None
    
    def test_regime_components_exported(self):
        """Test Regime components are exported in cognitive/__init__.py."""
        from iot_machine_learning.infrastructure.ml.cognitive import RegimeDetectionPipeline, OperationalRegimeClassifier
        
        # Verify components are importable
        assert RegimeDetectionPipeline is not None
        assert OperationalRegimeClassifier is not None
    
    def test_pattern_interpreter_integration(self):
        """Test PatternInterpreter is integrated in orchestrator."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import MetaCognitiveOrchestrator
        from iot_machine_learning.infrastructure.ml.cognitive import PatternInterpreter
        
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
        
        # Create orchestrator with pattern interpreter
        pattern_interpreter = PatternInterpreter()
        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            pattern_interpreter=pattern_interpreter,
        )
        
        # Verify pattern interpreter is set
        assert orchestrator._pattern_interpreter is not None
        assert isinstance(orchestrator._pattern_interpreter, PatternInterpreter)
    
    def test_rolling_window_engine_integration(self):
        """Test RollingWindowEngine is integrated in orchestrator."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import MetaCognitiveOrchestrator
        from iot_machine_learning.infrastructure.ml.cognitive import RollingWindowEngine
        
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
        
        # Create orchestrator with rolling window engine
        rolling_window_engine = RollingWindowEngine()
        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            rolling_window_engine=rolling_window_engine,
        )
        
        # Verify rolling window engine is set
        assert orchestrator._rolling_window_engine is not None
        assert isinstance(orchestrator._rolling_window_engine, RollingWindowEngine)
    
    def test_regime_detection_pipeline_integration(self):
        """Test RegimeDetectionPipeline is integrated in orchestrator."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.orchestrator import MetaCognitiveOrchestrator
        from iot_machine_learning.infrastructure.ml.cognitive import RegimeDetectionPipeline
        
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
        
        # Create orchestrator with regime detection pipeline
        regime_detection_pipeline = RegimeDetectionPipeline()
        orchestrator = MetaCognitiveOrchestrator(
            engines=[mock_engine],
            regime_detection_pipeline=regime_detection_pipeline,
        )
        
        # Verify regime detection pipeline is set
        assert orchestrator._regime_detection_pipeline is not None
        assert isinstance(orchestrator._regime_detection_pipeline, RegimeDetectionPipeline)
    
    def test_pipeline_context_has_all_components(self):
        """Test PipelineContext includes all medium-high impact components."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.context import create_initial_context
        from iot_machine_learning.infrastructure.ml.cognitive import (
            PatternInterpreter,
            RollingWindowEngine,
            RegimeDetectionPipeline,
        )
        
        # Create mock orchestrator
        mock_orchestrator = Mock()
        mock_orchestrator._metrics_collector = None
        mock_orchestrator._memory_health_monitor = None
        mock_orchestrator._memory_registry = None
        mock_orchestrator._neural_engine = None
        mock_orchestrator._decision_engine = None
        mock_orchestrator._pattern_interpreter = PatternInterpreter()
        mock_orchestrator._rolling_window_engine = RollingWindowEngine()
        mock_orchestrator._regime_detection_pipeline = RegimeDetectionPipeline()
        
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
        assert ctx.pattern_interpreter is not None
        assert ctx.rolling_window_engine is not None
        assert ctx.regime_detection_pipeline is not None
    
    def test_pattern_phase_exists(self):
        """Test PatternPhase is created and can be instantiated."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.pattern_phase import PatternPhase
        
        # Create phase
        phase = PatternPhase()
        
        # Verify phase exists and has correct name
        assert phase.name == "pattern"
    
    def test_dynamic_phase_exists(self):
        """Test DynamicPhase is created and can be instantiated."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.dynamic_phase import DynamicPhase
        
        # Create phase
        phase = DynamicPhase()
        
        # Verify phase exists and has correct name
        assert phase.name == "dynamic"
    
    def test_regime_phase_exists(self):
        """Test RegimePhase is created and can be instantiated."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.regime_phase import RegimePhase
        
        # Create phase
        phase = RegimePhase()
        
        # Verify phase exists and has correct name
        assert phase.name == "regime"
    
    def test_all_new_phases_in_pipeline_executor(self):
        """Test all new phases are included in default pipeline."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor import PipelineExecutor
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.pattern_phase import PatternPhase
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.dynamic_phase import DynamicPhase
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.regime_phase import RegimePhase
        
        # Create pipeline executor with default phases
        executor = PipelineExecutor()
        
        # Verify all new phases are in the pipeline
        phase_names = [phase.name for phase in executor._phases]
        assert "pattern" in phase_names
        assert "dynamic" in phase_names
        assert "regime" in phase_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
