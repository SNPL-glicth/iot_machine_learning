"""Test: Verificar orden correcto de fases DecisionArbiterPhase → FusePhase.

Este test prueba que:
1. DecisionArbiterPhase corre ANTES de FusePhase
2. DecisionArbiterPhase setea ctx.selected_engine
3. FusePhase respeta ctx.selected_engine pre-definido

Audit Issue: DecisionArbiterPhase corría DESPUÉS de FusePhase, haciendo
que no pudiera cambiar el motor seleccionado.
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


class TestPipelinePhaseOrder:
    """Tests para verificar orden correcto de fases."""
    
    def test_decision_arbiter_phase_comes_before_fuse_phase(self):
        """Test: DecisionArbiterPhase debe correr antes de FusePhase."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.pipeline_executor import (
            PipelineExecutor,
        )
        
        executor = PipelineExecutor()
        
        # Obtener nombres de fases en orden
        phase_names = [p.name for p in executor._phases]
        
        # Verificar que DecisionArbiterPhase viene ANTES de FusePhase
        arbiter_idx = phase_names.index("decision_arbiter")
        fuse_idx = phase_names.index("fuse")
        
        assert arbiter_idx < fuse_idx, (
            f"DecisionArbiterPhase (index {arbiter_idx}) debe correr "
            f"ANTES de FusePhase (index {fuse_idx})"
        )
    
    def test_decision_arbiter_sets_selected_engine(self):
        """Test: DecisionArbiterPhase setea ctx.selected_engine cuando está habilitado.
        
        Nota: Según EngineDecisionArbiter rules, si flag_engine != profile_engine,
        el profile_engine gana (Rule 3).
        """
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.decision_arbiter_phase import (
            DecisionArbiterPhase,
        )
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import PipelineContext
        
        # Crear contexto mock con perceptions
        mock_flags = MagicMock()
        mock_flags.ML_DECISION_ARBITER_ENABLED = True
        mock_flags.ML_ROLLBACK_TO_BASELINE = False
        # flag_engine es "taylor" pero profile es "baseline_moving_average" -> profile gana
        mock_flags.get_active_engine_for_series.return_value = "taylor"
        mock_flags.ML_ENGINE_SERIES_OVERRIDES = {}
        
        ctx = PipelineContext(
            orchestrator=MagicMock(),
            values=[1.0, 2.0, 3.0],
            timestamps=None,
            series_id="test_series",
            flags=mock_flags,
            timer=MagicMock(),
            perceptions=[
                MockPerception("baseline_moving_average"),  # Este es profile_engine
                MockPerception("taylor"),
            ],
            selected_engine=None,  # No pre-set
        )
        
        phase = DecisionArbiterPhase()
        new_ctx = phase.execute(ctx)
        
        # Verificar que selected_engine fue seteado
        assert new_ctx.selected_engine is not None, (
            "DecisionArbiterPhase debe setear ctx.selected_engine"
        )
        # Como flag("taylor") != profile("baseline_moving_average"), profile gana (Rule 3)
        assert new_ctx.selected_engine == "baseline_moving_average", (
            f"Esperado 'baseline_moving_average' (profile wins), obtenido '{new_ctx.selected_engine}'"
        )
        assert new_ctx.engine_decision is not None, (
            "DecisionArbiterPhase debe setear ctx.engine_decision"
        )
    
    def test_decision_arbiter_sets_default_when_disabled(self):
        """Test: Cuando arbiter está deshabilitado, aún setea default desde perceptions."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.decision_arbiter_phase import (
            DecisionArbiterPhase,
        )
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import PipelineContext
        
        mock_flags = MagicMock()
        mock_flags.ML_DECISION_ARBITER_ENABLED = False  # Deshabilitado
        
        ctx = PipelineContext(
            orchestrator=MagicMock(),
            values=[1.0, 2.0, 3.0],
            timestamps=None,
            series_id="test_series",
            flags=mock_flags,
            timer=MagicMock(),
            perceptions=[
                MockPerception("baseline_moving_average"),
            ],
            selected_engine=None,
        )
        
        phase = DecisionArbiterPhase()
        new_ctx = phase.execute(ctx)
        
        # Aún cuando arbiter está deshabilitado, debe setear default
        assert new_ctx.selected_engine is not None, (
            "DecisionArbiterPhase debe setear default selected_engine "
            "aunque esté deshabilitado"
        )
        assert new_ctx.selected_engine == "baseline_moving_average"
    
    def test_fuse_phase_uses_preselected_engine(self):
        """Test: FusePhase usa ctx.selected_engine pre-definido por arbiter."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import FusePhase
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import PipelineContext
        
        mock_orchestrator = MagicMock()
        mock_orchestrator._fusion = MagicMock()
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
                MockPerception("baseline_moving_average", predicted_value=1.0),
                MockPerception("taylor", predicted_value=2.5),  # Pre-selected
            ],
            selected_engine="taylor",  # Pre-set by arbiter
            inhibition_states=[],
        )
        
        phase = FusePhase()
        new_ctx = phase.execute(ctx)
        
        # Verificar que FusePhase usó el motor pre-seleccionado
        assert new_ctx.selected_engine == "taylor", (
            f"FusePhase debe usar selected_engine pre-definido. "
            f"Esperado 'taylor', obtenido '{new_ctx.selected_engine}'"
        )
        assert new_ctx.fused_value == 2.5, (
            f"FusePhase debe usar valor del motor pre-seleccionado. "
            f"Esperado 2.5, obtenido {new_ctx.fused_value}"
        )
        assert new_ctx.fusion_method == "arbiter_override", (
            f"Esperado method='arbiter_override', obtenido '{new_ctx.fusion_method}'"
        )
        
        # Verificar que fusion.fuse() NO fue llamado (override path)
        mock_orchestrator._fusion.fuse.assert_not_called()
    
    def test_fuse_phase_falls_back_to_weighted_fusion_when_no_preselection(self):
        """Test: FusePhase usa weighted fusion cuando no hay pre-selection."""
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases.fuse_phase import FusePhase
        from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phases import PipelineContext
        
        mock_fusion = MagicMock()
        mock_fusion.fuse.return_value = (1.5, 0.8, "flat", {"baseline": 0.6, "taylor": 0.4}, "baseline", "weighted")
        
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
                MockPerception("baseline_moving_average"),
                MockPerception("taylor"),
            ],
            selected_engine=None,  # No pre-selection
            inhibition_states=[],
        )
        
        phase = FusePhase()
        new_ctx = phase.execute(ctx)
        
        # Verificar que fusion.fuse() fue llamado (default path)
        mock_fusion.fuse.assert_called_once()
        assert new_ctx.fusion_method == "weighted_average"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
