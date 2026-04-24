"""Test: Semantic Weight Modifier en FusePhase.

Verifica que urgency_score, sentiment e impact_score afectan
los pesos de fusión de engines (no solo la severidad).
"""

import pytest
from unittest.mock import MagicMock


class MockPerception:
    """Mock perception para testing."""
    def __init__(self, engine_name, predicted_value=1.0, confidence=0.8, trend="flat"):
        self.engine_name = engine_name
        self.predicted_value = predicted_value
        self.confidence = confidence
        self.trend = trend


class TestSemanticWeightModifier:
    """Tests para verificar modificador semántico de pesos."""
    
    def test_urgency_high_boosts_taylor_weight(self):
        """Test: urgency_score > 0.7 aumenta peso de taylor_engine en 15%."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import FusePhase
        
        phase = FusePhase()
        
        # Pesos originales
        weights = {
            "baseline_moving_average": 0.5,
            "taylor": 0.5,
        }
        
        # Contexto con urgency_score alto
        ctx = MagicMock()
        ctx.series_id = "test_series"
        ctx.metadata = {"urgency_score": 0.85}  # > 0.7
        ctx.perceptions = [
            MockPerception("baseline_moving_average"),
            MockPerception("taylor"),
        ]
        ctx.urgency_score = None  # Not set as attribute, only in metadata
        ctx.sentiment_label = None
        ctx.impact_score = None
        
        modified = phase._apply_semantic_modifier(weights, ctx)
        
        # Verificar que taylor recibió boost de 15%
        assert modified["taylor"] == 0.65, (
            f"taylor debería tener 0.5 + 0.15 = 0.65, tiene {modified['taylor']}"
        )
        # Baseline no cambió
        assert modified["baseline_moving_average"] == 0.5
    
    def test_urgency_low_boosts_baseline_weight(self):
        """Test: urgency_score < 0.3 aumenta peso de baseline en 15%."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import FusePhase
        
        phase = FusePhase()
        
        weights = {
            "baseline_moving_average": 0.5,
            "taylor": 0.5,
        }
        
        ctx = MagicMock()
        ctx.series_id = "test_series"
        ctx.metadata = {"urgency_score": 0.2}  # < 0.3
        ctx.perceptions = [
            MockPerception("baseline_moving_average"),
            MockPerception("taylor"),
        ]
        ctx.urgency_score = None
        ctx.sentiment_label = None
        ctx.impact_score = None
        
        modified = phase._apply_semantic_modifier(weights, ctx)
        
        # Verificar que baseline recibió boost de 15%
        assert modified["baseline_moving_average"] == 0.65, (
            f"baseline debería tener 0.5 + 0.15 = 0.65, tiene {modified['baseline_moving_average']}"
        )
        # Taylor no cambió
        assert modified["taylor"] == 0.5
    
    def test_negative_sentiment_high_impact_penalizes_low_confidence(self):
        """Test: sentiment negative + impact > 0.5 reduce peso de engines con confianza baja."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import FusePhase
        
        phase = FusePhase()
        
        weights = {
            "baseline_moving_average": 0.5,
            "low_conf_engine": 0.5,
        }
        
        ctx = MagicMock()
        ctx.series_id = "test_series"
        ctx.metadata = {
            "sentiment_label": "negative",
            "impact_score": 0.6,  # > 0.5
        }
        # Engine con confianza baja (< 0.4)
        ctx.perceptions = [
            MockPerception("baseline_moving_average", confidence=0.8),
            MockPerception("low_conf_engine", confidence=0.3),  # < 0.4
        ]
        ctx.urgency_score = None
        ctx.sentiment_label = None
        ctx.impact_score = None
        
        modified = phase._apply_semantic_modifier(weights, ctx)
        
        # Verificar que engine de baja confianza fue penalizado (20% reduction)
        assert modified["low_conf_engine"] == 0.4, (
            f"low_conf_engine debería tener 0.5 * 0.8 = 0.4, tiene {modified['low_conf_engine']}"
        )
        # Baseline no cambió (tiene alta confianza)
        assert modified["baseline_moving_average"] == 0.5
    
    def test_no_semantic_signals_returns_unchanged_weights(self):
        """Test: Sin señales semánticas (pipeline IoT), pesos sin cambio."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import FusePhase
        
        phase = FusePhase()
        
        weights = {
            "baseline_moving_average": 0.5,
            "taylor": 0.5,
        }
        
        ctx = MagicMock()
        ctx.series_id = "test_series"
        ctx.metadata = {}  # Sin señales semánticas
        ctx.perceptions = [
            MockPerception("baseline_moving_average"),
            MockPerception("taylor"),
        ]
        ctx.urgency_score = None
        ctx.sentiment_label = None
        ctx.impact_score = None
        
        modified = phase._apply_semantic_modifier(weights, ctx)
        
        # Pesos deben ser idénticos
        assert modified == weights, (
            f"Sin señales semánticas, pesos deben ser iguales. "
            f"Original: {weights}, Modificado: {modified}"
        )
    
    def test_urgency_mid_range_no_change(self):
        """Test: urgency_score entre 0.3 y 0.7 no causa cambios."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import FusePhase
        
        phase = FusePhase()
        
        weights = {
            "baseline_moving_average": 0.5,
            "taylor": 0.5,
        }
        
        ctx = MagicMock()
        ctx.series_id = "test_series"
        ctx.metadata = {"urgency_score": 0.5}  # Entre 0.3 y 0.7
        ctx.perceptions = [
            MockPerception("baseline_moving_average"),
            MockPerception("taylor"),
        ]
        ctx.urgency_score = None
        ctx.sentiment_label = None
        ctx.impact_score = None
        
        modified = phase._apply_semantic_modifier(weights, ctx)
        
        # Sin cambios
        assert modified == weights
    
    def test_semantic_signals_from_context_attributes(self):
        """Test: Señales semánticas pueden venir de atributos de contexto (no solo metadata)."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import FusePhase
        
        phase = FusePhase()
        
        weights = {
            "baseline_moving_average": 0.5,
            "taylor": 0.5,
        }
        
        # Contexto con urgency_score como atributo directo
        ctx = MagicMock()
        ctx.series_id = "test_series"
        ctx.metadata = {}
        ctx.perceptions = [
            MockPerception("baseline_moving_average"),
            MockPerception("taylor"),
        ]
        # Set as context attributes directly (simulating text pipeline)
        ctx.urgency_score = 0.85  # > 0.7
        ctx.sentiment_label = None
        ctx.impact_score = None
        
        modified = phase._apply_semantic_modifier(weights, ctx)
        
        # Taylor debe recibir boost
        assert modified["taylor"] == 0.65, (
            f"taylor debería tener 0.5 + 0.15 = 0.65, tiene {modified['taylor']}"
        )
    
    def test_fusion_recomputation_with_modified_weights(self):
        """Test: FusePhase re-calcula fusión cuando pesos son modificados."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import FusePhase
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import PipelineContext
        
        mock_fusion = MagicMock()
        # First call returns equal weights, second call after modification
        mock_fusion.fuse.return_value = (
            1.5,  # fused_val
            0.7,  # fused_conf
            "flat",  # fused_trend
            {"baseline_moving_average": 0.5, "taylor": 0.5},  # final_weights
            "baseline_moving_average",  # selected
            "weighted_average",  # reason
        )
        
        mock_orchestrator = MagicMock()
        mock_orchestrator._fusion = mock_fusion
        mock_orchestrator._correlation_port = None
        
        mock_flags = MagicMock()
        
        ctx = PipelineContext(
            orchestrator=mock_orchestrator,
            values=[1.0, 2.0, 3.0],
            timestamps=None,
            series_id="test_series",
            flags=mock_flags,
            timer=MagicMock(),
            perceptions=[
                MockPerception("baseline_moving_average", predicted_value=1.0, confidence=0.8),
                MockPerception("taylor", predicted_value=2.0, confidence=0.7),
            ],
            selected_engine=None,  # No pre-selection
            inhibition_states=[],
            metadata={"urgency_score": 0.85},  # High urgency
        )
        
        phase = FusePhase()
        new_ctx = phase.execute(ctx)
        
        # Verificar que los pesos fueron modificados
        # Taylor debería tener más peso por urgency alta
        assert new_ctx.final_weights["taylor"] > new_ctx.final_weights["baseline_moving_average"], (
            f"Con urgency alta, taylor debe tener más peso. "
            f"Pesos: {new_ctx.final_weights}"
        )
        
        # Verificar que la razón indica ajuste semántico
        assert "semantic_weight_adjusted" in new_ctx.selection_reason, (
            f"Reason debe indicar ajuste semántico: {new_ctx.selection_reason}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
