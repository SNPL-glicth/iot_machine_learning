"""Unit tests para RosaRojaMoEEngine como cabeza MoE y PredictionEngine estándar."""

from __future__ import annotations

import numpy as np
import pytest

from iot_machine_learning.infrastructure.ml.engines.core.factory import EngineFactory
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.engine import RosaRojaMoEEngine
from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionResult,
)
from iot_machine_learning.infrastructure.ml.moe.expert_wrappers.engine_adapter import (
    create_rosa_roja_expert,
)
from iot_machine_learning.infrastructure.ml.moe.experts.rosa_roja_expert import (
    RosaRojaResult,
)
from iot_machine_learning.domain.entities.iot.sensor_reading import (
    SensorWindow,
    Reading,
)


@pytest.fixture
def synthetic_series() -> list[float]:
    """Genera serie sintética con tendencia alcista suave (30 puntos)."""
    np.random.seed(42)
    base = 100.0
    return [base + i * 0.5 + np.random.normal(0, 0.05) for i in range(30)]


class TestRosaRojaMoEEngine:
    """Verificación de contrato PredictionEngine y MoE Head."""

    def test_engine_registration_and_creation(self) -> None:
        """RosaRojaMoEEngine se registra en EngineFactory y se crea por nombre."""
        assert "rosa_roja" in EngineFactory.list_engines()
        engine = EngineFactory.create("rosa_roja")
        assert isinstance(engine, RosaRojaMoEEngine)
        assert engine.name == "rosa_roja"

    def test_can_handle_threshold(self) -> None:
        """Requiere un mínimo de puntos para inicializar trayectorias."""
        engine = RosaRojaMoEEngine()
        assert not engine.can_handle(5)
        assert not engine.can_handle(10)
        assert engine.can_handle(11)
        assert engine.can_handle(30)

    def test_predict_insufficient_data(self) -> None:
        """Devuelve resultado seguro con confianza 0 si hay datos insuficientes."""
        engine = RosaRojaMoEEngine()
        res = engine.predict([100.0, 101.0, 102.0])
        assert isinstance(res, PredictionResult)
        assert res.confidence == 0.0
        assert res.trend == "stable"
        assert res.metadata.get("reason") == "insufficient_history"

    def test_predict_full_pipeline(self, synthetic_series: list[float]) -> None:
        """Ejecuta el ciclo cognitivo MoE completo y retorna PredictionResult."""
        engine = RosaRojaMoEEngine()
        res = engine.predict(synthetic_series)

        assert isinstance(res, PredictionResult)
        assert isinstance(res.predicted_value, float)
        assert 0.0 <= res.confidence <= 1.0
        assert res.trend in ("up", "down", "stable")
        assert res.metadata["engine"] == "rosa_roja_moe_head"
        assert "action" in res.metadata

    def test_analyze_contract_compatibility(self, synthetic_series: list[float]) -> None:
        """El método analyze() cumple la interfaz esperada por RosaRojaExpert."""
        engine = RosaRojaMoEEngine()
        s_t = {
            "values": synthetic_series,
            "timestamps": [float(i) for i in range(len(synthetic_series))],
            "current_regime": "trending",
        }
        result = engine.analyze(s_t)

        assert isinstance(result, RosaRojaResult)
        assert result.expected_direction in ("up", "down", "stable")
        assert 0.0 <= result.confidence <= 1.0
        assert result.status in ("ok", "hold")

    def test_expert_adapter_with_real_engine(self, synthetic_series: list[float]) -> None:
        """RosaRojaExpert consume RosaRojaMoEEngine real sin fallbacks de error."""
        engine = EngineFactory.create("rosa_roja")
        expert = create_rosa_roja_expert(
            engine=engine, min_history_points=11, enabled=True
        )

        window = SensorWindow(
            series_id="TEST_SERIES",
            readings=[
                Reading(series_id="TEST_SERIES", value=v, timestamp=float(i))
                for i, v in enumerate(synthetic_series)
            ],
        )
        assert expert.can_handle(window)
        output = expert.predict(window)

        assert output.confidence >= 0.0
        assert output.trend in ("up", "down", "stable")
        assert output.metadata["engine_name"] == "rosa_roja"
        assert output.metadata["rosa_roja_status"] != "unavailable"

    def test_as_port_bridge_compatibility(self, synthetic_series: list[float]) -> None:
        """El engine se envuelve limpiamente como PredictionPort vía as_port()."""
        engine = EngineFactory.create("rosa_roja")
        port = engine.as_port()
        assert port.name == "rosa_roja"
