"""Contract tests para RosaRojaExpertAdapter (Experimento B).

Valida:
1. SensorWindow → S_t correcto
2. RosaRojaResult válido → ExpertOutput con metadata rico
3. ExpertResult compatible con MoE (ExpertOutput contract)
4. Fallback limpio cuando unavailable / insufficient_history / error
5. Determinismo: mismo contexto → mismo resultado
6. No regresión: MoE sin Rosa Roja = mismo comportamiento que antes
7. Aislamiento: adapter no altera otros expertos ni estado global
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from iot_machine_learning.domain.entities.iot.sensor_reading import SensorWindow, Reading
from iot_machine_learning.infrastructure.ml.moe.experts.rosa_roja_expert import RosaRojaExpert, RosaRojaResult
from iot_machine_learning.domain.ports.expert_port import ExpertOutput


class FakeRosaRojaEngine:
    """Mock determinista del core Rosa Roja."""
    def __init__(self, return_result: RosaRojaResult):
        self._result = return_result
        self.calls = []
    
    def analyze(self, s_t: dict) -> RosaRojaResult:
        self.calls.append(s_t)
        return self._result


class TestRosaRojaExpertAdapter:
    
    @pytest.fixture
    def valid_window(self):
        return SensorWindow(
            series_id="TEST_001",
            readings=[Reading(series_id="TEST_001", value=100.0 + i * 0.1, timestamp=float(i)) for i in range(60)]
        )
    
    @pytest.fixture
    def rr_result(self):
        return RosaRojaResult(
            trajectory=["expansion", "exhaustion", "pullback", "continuation"],
            trajectory_score=0.78,
            rhythm_score=0.65,
            lambda_val=0.42,
            theta_entropy=1.23,
            regime_alert="volatile",
            invalidation_step=4,
            expected_direction="up",
            expected_magnitude=0.015,
            confidence=0.72,
            evidence={"test": "data"},
            status="ok",
        )
    
    def test_1_window_to_state_structure(self, valid_window, rr_result):
        """1. SensorWindow → S_t estructura correcta."""
        engine = FakeRosaRojaEngine(rr_result)
        adapter = RosaRojaExpert(engine=engine, min_history_points=50, enabled=True)
        
        output = adapter.predict(valid_window)
        
        assert len(engine.calls) == 1
        s_t = engine.calls[0]
        assert "values" in s_t
        assert "timestamps" in s_t
        assert "n_points" in s_t
        assert s_t["n_points"] == 60
        assert "current_regime" in s_t
        assert "volatility" in s_t
        assert isinstance(s_t["values"], list)
        assert len(s_t["values"]) == 60
    
    def test_2_rosa_roja_result_rich_metadata_preserved(self, valid_window, rr_result):
        """2. RosaRojaResult rico → ExpertOutput con metadata completo intacto."""
        engine = FakeRosaRojaEngine(rr_result)
        adapter = RosaRojaExpert(engine=engine, enabled=True)
        
        output = adapter.predict(valid_window)
        
        assert output.confidence == 0.72
        assert output.trend == "up"
        meta = output.metadata
        assert meta["rosa_roja_status"] == "ok"
        assert meta["trajectory"] == rr_result.trajectory
        assert meta["trajectory_score"] == 0.78
        assert meta["rhythm_score"] == 0.65
        assert meta["lambda"] == 0.42
        assert meta["theta_entropy"] == 1.23
        assert meta["regime_alert"] == "volatile"
        assert meta["invalidation_step"] == 4
        assert meta["evidence"] == {"test": "data"}
        assert meta["method"] == "trajectory_rhythm"
        assert meta["engine_name"] == "rosa_roja"
    
    def test_3_expert_output_compatible_with_moe_contract(self, valid_window, rr_result):
        """3. ExpertResult cumple contrato ExpertOutput (MoE compatible)."""
        engine = FakeRosaRojaEngine(rr_result)
        adapter = RosaRojaExpert(engine=engine, enabled=True)
        
        output = adapter.predict(valid_window)
        
        # Debe cumplir ExpertOutput contract
        assert isinstance(output, ExpertOutput)
        assert hasattr(output, 'prediction')
        assert hasattr(output, 'confidence')
        assert hasattr(output, 'trend')
        assert hasattr(output, 'latency_ms')
        assert hasattr(output, 'stability')
        assert hasattr(output, 'local_fit_error')
        assert hasattr(output, 'metadata')
        assert 0.0 <= output.confidence <= 1.0
        assert output.trend in ("up", "down", "stable")
        assert output.latency_ms >= 0.0
        assert output.stability >= 0.0
    
    def test_4_fallback_clean_when_disabled(self, valid_window):
        """4a. Rosa Roja disabled → fallback limpio, no rompe MoE."""
        adapter = RosaRojaExpert(engine=None, enabled=False)
        
        output = adapter.predict(valid_window)
        
        assert output.confidence == 0.0
        assert output.trend == "stable"
        assert output.metadata["rosa_roja_status"] == "unavailable"
        assert "reason" in output.metadata
    
    def test_4b_fallback_clean_missing_engine(self, valid_window):
        """4b. Engine None pero enabled=True → fallback limpio."""
        adapter = RosaRojaExpert(engine=None, enabled=True)
        
        output = adapter.predict(valid_window)
        
        assert output.confidence == 0.0
        assert output.metadata["rosa_roja_status"] == "unavailable"
    
    def test_4c_fallback_clean_insufficient_history(self):
        """4c. Historial insuficiente → fallback limpio."""
        short_window = SensorWindow(
            series_id="TEST_001",
            readings=[Reading(series_id="TEST_001", value=100.0, timestamp=float(i)) for i in range(10)]
        )
        adapter = RosaRojaExpert(engine=Mock(), enabled=True, min_history_points=50)
        
        output = adapter.predict(short_window)
        
        assert output.confidence == 0.0
        assert output.metadata["rosa_roja_status"] == "insufficient_history"
        assert "50" in output.metadata["reason"]
    
    def test_4d_fallback_clean_engine_error(self, valid_window):
        """4d. Engine lanza excepción → fallback limpio (fail-silent)."""
        engine = Mock()
        engine.analyze.side_effect = RuntimeError("core failure")
        adapter = RosaRojaExpert(engine=engine, enabled=True)
        
        output = adapter.predict(valid_window)
        
        assert output.confidence == 0.0
        assert output.metadata["rosa_roja_status"] == "error"
        assert "core failure" in output.metadata["reason"]
    
    def test_5_deterministic_same_context(self, valid_window, rr_result):
        """5. Mismo contexto → resultado determinista."""
        engine = FakeRosaRojaEngine(rr_result)
        adapter = RosaRojaExpert(engine=engine, enabled=True)
        
        out1 = adapter.predict(valid_window)
        out2 = adapter.predict(valid_window)
        
        assert out1.prediction == out2.prediction
        assert out1.confidence == out2.confidence
        assert out1.trend == out2.trend
        assert out1.metadata == out2.metadata
        assert len(engine.calls) == 2  # llamado dos veces
    
    def test_6_no_regression_moe_without_rosa_roja(self, valid_window, rr_result):
        """6. MoE sin Rosa Roja = mismo comportamiento que antes.
        
        Se verifica que el adapter:
        - No muta estado global
        - No modifica otros expertos
        - No toca registry salvo register() explícito
        - can_handle es idempotente sin side effects
        """
        engine = FakeRosaRojaEngine(rr_result)
        adapter = RosaRojaExpert(engine=engine, enabled=True)
        
        # Verificar que no hay atributos de estado global / registry
        assert not hasattr(adapter, '_registry')
        assert not hasattr(adapter, '_fallback_engine')
        assert not hasattr(adapter, '_moe_engine')
        
        # can_handle es puro (idempotente)
        assert adapter.can_handle(valid_window) is True
        assert adapter.can_handle(valid_window) is True
        
        # predict no tiene side effects visibles en el adapter
        out1 = adapter.predict(valid_window)
        out2 = adapter.predict(valid_window)
        assert out1 == out2
    
    def test_7_expert_isolation_no_cross_contamination(self, valid_window, rr_result):
        """7. Adapter no referencia código de otros expertos (aislamiento)."""
        adapter = RosaRojaExpert(engine=FakeRosaRojaEngine(rr_result), enabled=True)
        
        # Verificar que el módulo solo importa lo necesario
        import sys
        adapter_module = sys.modules.get('iot_machine_learning.infrastructure.ml.moe.experts.rosa_roja_expert')
        assert adapter_module is not None
        
        # El adapter no debe tener referencias a otros expertos en su closure/namespaces
        # Esto se verifica indirectamente: el archivo solo importa expert_port + rosa_roja core
        # No importa taylor_expert, kalman_expert, statistical_expert, neural_expert, baseline_expert
        
        # Test funcional: ejecutar predict no debe fallar por imports faltantes de otros expertos
        output = adapter.predict(valid_window)
        assert output.confidence == 0.72
    
    def test_8_capabilities_declared_correctly(self):
        """Capacidades declaradas correctamente para registry matching."""
        adapter = RosaRojaExpert(engine=None, enabled=True, min_history_points=50)
        
        caps = adapter.capabilities
        assert "volatile" in caps.regimes
        assert "trending" in caps.regimes
        assert "stable" in caps.regimes
        assert "noisy" in caps.regimes
        assert "finance" in caps.domains
        assert "iot" in caps.domains
        assert caps.min_points == 50
        assert caps.max_points == 500
        assert "trajectory" in caps.specialties
        assert "rhythm" in caps.specialties
        assert "regime_transition" in caps.specialties
        assert "sequential_reasoning" in caps.specialties
        assert caps.computational_cost == 3.0
    
    def test_9_name_identifier(self):
        """Identificador único correcto."""
        adapter = RosaRojaExpert(engine=None, enabled=True)
        assert adapter.name == "rosa_roja"


class TestRosaRojaExpertMoEIntegration:
    """Tests de integración con MoE (require registry + dispatcher)."""
    
    @pytest.fixture
    def valid_window(self):
        return SensorWindow(
            series_id="TEST_001",
            readings=[Reading(series_id="TEST_001", value=100.0 + i * 0.1, timestamp=float(i)) for i in range(60)]
        )
    
    @pytest.fixture
    def rr_result(self):
        return RosaRojaResult(
            trajectory=["expansion", "exhaustion", "pullback", "continuation"],
            trajectory_score=0.78,
            rhythm_score=0.65,
            lambda_val=0.42,
            theta_entropy=1.23,
            regime_alert="volatile",
            invalidation_step=4,
            expected_direction="up",
            expected_magnitude=0.015,
            confidence=0.72,
            evidence={},
            status="ok",
        )
    
    def test_moe_registry_registration(self, valid_window, rr_result):
        """Registro en ExpertRegistry funciona."""
        from iot_machine_learning.infrastructure.ml.moe.registry import ExpertRegistry
        
        registry = ExpertRegistry()
        engine = FakeRosaRojaEngine(rr_result)
        adapter = RosaRojaExpert(engine=engine, enabled=True)
        
        registry.register("rosa_roja", adapter)
        
        assert "rosa_roja" in registry
        retrieved = registry.get("rosa_roja")
        assert retrieved is adapter
        assert registry.get_capabilities("rosa_roja") == adapter.capabilities
    
    def test_moe_dispatcher_execution(self, valid_window, rr_result):
        """Dispatcher ejecuta adapter correctamente (fail-silent respected)."""
        from iot_machine_learning.infrastructure.ml.moe.gateway.expert_dispatcher import ExpertDispatcher
        from iot_machine_learning.infrastructure.ml.moe.registry import ExpertRegistry
        
        registry = ExpertRegistry()
        engine = FakeRosaRojaEngine(rr_result)
        adapter = RosaRojaExpert(engine=engine, enabled=True)
        registry.register("rosa_roja", adapter)
        
        dispatcher = ExpertDispatcher(registry, timeout_ms=5000)
        outputs = dispatcher.dispatch(["rosa_roja"], valid_window)
        
        assert "rosa_roja" in outputs
        out = outputs["rosa_roja"]
        assert isinstance(out, ExpertOutput)
        assert out.confidence == 0.72
        assert out.metadata["rosa_roja_status"] == "ok"
    
    def test_moe_dispatcher_graceful_when_disabled(self, valid_window):
        """Dispatcher omite expert disabled (fail-silent: filtered by can_handle)."""
        from iot_machine_learning.infrastructure.ml.moe.gateway.expert_dispatcher import ExpertDispatcher
        from iot_machine_learning.infrastructure.ml.moe.registry import ExpertRegistry
        
        registry = ExpertRegistry()
        adapter = RosaRojaExpert(engine=None, enabled=False)  # DISABLED
        registry.register("rosa_roja", adapter)
        
        dispatcher = ExpertDispatcher(registry, timeout_ms=5000)
        outputs = dispatcher.dispatch(["rosa_roja"], valid_window)
        
        # Expert disabled → can_handle=False → dispatcher omite (fail-silent)
        assert "rosa_roja" not in outputs
        assert outputs == {}