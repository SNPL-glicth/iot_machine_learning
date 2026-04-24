"""Test: Degradation tracking in predictions.

Verifica que silent fallbacks son visibles via:
- degradation_reasons: list[str]
- is_degraded: bool
- Logging de DEGRADED_PREDICTION
"""

import pytest
from unittest.mock import MagicMock


class MockInhibitionState:
    """Mock inhibition state."""
    def __init__(self, engine_name, inhibited_weight, suppression_factor=0.0):
        self.engine_name = engine_name
        self.inhibited_weight = inhibited_weight
        self.suppression_factor = suppression_factor


class TestDegradationTracking:
    """Tests para verificar tracking de degradación."""
    
    def test_prediction_result_has_degradation_fields(self):
        """Test: PredictionResult tiene degradation_reasons e is_degraded."""
        from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult
        
        result = PredictionResult(
            predicted_value=1.0,
            confidence=0.8,
            trend="stable",
            degradation_reasons=("budget_exceeded",),
            is_degraded=True,
        )
        
        assert result.is_degraded is True
        assert "budget_exceeded" in result.degradation_reasons
    
    def test_prediction_result_default_not_degraded(self):
        """Test: Por defecto, PredictionResult no está degradado."""
        from iot_machine_learning.infrastructure.ml.interfaces import PredictionResult
        
        result = PredictionResult(
            predicted_value=1.0,
            confidence=0.8,
            trend="stable",
        )
        
        assert result.is_degraded is False
        assert len(result.degradation_reasons) == 0
    
    def test_pipeline_context_has_degradation_reasons(self):
        """Test: PipelineContext puede almacenar degradation reasons."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import (
            PipelineContext,
        )
        
        ctx = PipelineContext(
            orchestrator=MagicMock(),
            values=[1.0, 2.0, 3.0],
            timestamps=None,
            series_id="test_series",
            flags=MagicMock(),
            timer=MagicMock(),
            degradation_reasons=["moe_init_failed"],
        )
        
        assert "moe_init_failed" in ctx.degradation_reasons
    
    def test_all_engines_inhibited_degradation(self):
        """Test: Cuando todos los engines están inhibidos, se registra degradación."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import (
            FusePhase,
        )
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import (
            PipelineContext,
        )
        
        class MockPerception:
            def __init__(self, engine_name, confidence=0.8):
                self.engine_name = engine_name
                self.predicted_value = 1.0
                self.confidence = confidence
                self.trend = "stable"
        
        mock_orchestrator = MagicMock()
        mock_orchestrator._fusion = MagicMock()
        mock_orchestrator._fusion.fuse.return_value = (
            1.0, 0.8, "stable", {"eng1": 0.5, "eng2": 0.5}, "eng1", "test"
        )
        mock_orchestrator._correlation_port = None
        
        # All engines inhibited (suppression_factor > 0.99)
        inhibition_states = [
            MockInhibitionState("eng1", 0.01, suppression_factor=0.995),
            MockInhibitionState("eng2", 0.01, suppression_factor=0.995),
        ]
        
        ctx = PipelineContext(
            orchestrator=mock_orchestrator,
            values=[1.0, 2.0, 3.0],
            timestamps=None,
            series_id="test_series",
            flags=MagicMock(),
            timer=MagicMock(),
            perceptions=[MockPerception("eng1"), MockPerception("eng2")],
            inhibition_states=inhibition_states,
            selected_engine=None,
        )
        
        phase = FusePhase()
        new_ctx = phase.execute(ctx)
        
        # Degradation reason should be present
        assert "all_engines_inhibited" in new_ctx.degradation_reasons
    
    def test_fallback_result_is_degraded(self):
        """Test: Fallback result está marcado como degradado."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor import (
            PipelineExecutor,
        )
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import (
            PipelineContext,
        )
        
        executor = PipelineExecutor()
        
        mock_orchestrator = MagicMock()
        mock_orchestrator._last_explanation = None
        
        ctx = PipelineContext(
            orchestrator=mock_orchestrator,
            values=[1.0],
            timestamps=None,
            series_id="test_series",
            flags=MagicMock(),
            timer=MagicMock(),
            is_fallback=True,
            fallback_reason="engine_failure",
            degradation_reasons=["test_reason"],
        )
        
        result = executor._create_fallback_result(ctx)
        
        assert result.is_degraded is True
        assert "fallback:engine_failure" in result.degradation_reasons
        assert "test_reason" in result.degradation_reasons
    
    def test_container_tracks_moe_init_failure(self):
        """Test: Container tracks MoE init failure as degradation."""
        from iot_machine_learning.ml_service.runners.wiring.container import (
            BatchEnterpriseContainer,
        )
        from unittest.mock import patch
        
        mock_flags = MagicMock()
        mock_flags.ML_MOE_ENABLED = True
        
        with patch("sqlalchemy.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine
            
            container = BatchEnterpriseContainer(
                engine=mock_engine,
                flags=mock_flags,
            )
            
            # Simulate MoE init failure
            with patch(
                "iot_machine_learning.ml_service.runners.wiring.container.create_moe_gateway_safe"
            ) as mock_create:
                mock_create.return_value = None  # MoE init failed
                
                gateway = container._get_or_create_moe_gateway()
                
                assert gateway is None
                assert "moe_init_failed" in container._degradation_reasons


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
