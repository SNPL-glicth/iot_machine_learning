"""Tests para EngineFactory y BaselineMovingAverageEngine.

Verifica:
- Registro y creación de motores
- Fallback a baseline cuando motor no existe
- Selección por feature flags (panic button, whitelist, overrides)
- BaselineMovingAverageEngine como wrapper del baseline existente
"""

from __future__ import annotations

import pytest

from iot_machine_learning.application.use_cases.select_engine import (
    select_engine_for_sensor,
)
from iot_machine_learning.infrastructure.ml.engines.core import (
    BaselineMovingAverageEngine,
    EngineFactory,
)
from iot_machine_learning.infrastructure.ml.interfaces import (
    PredictionEngine,
    PredictionEnginePortBridge,
    PredictionResult,
)
from iot_machine_learning.infrastructure.ml.engines.taylor import TaylorPredictionEngine
from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags


class _DummyEngine(PredictionEngine):
    """Motor dummy para testing."""

    @property
    def name(self) -> str:
        return "dummy"

    def can_handle(self, n_points: int) -> bool:
        return n_points >= 1

    def predict(self, values, timestamps=None):
        return PredictionResult(
            predicted_value=999.0,
            confidence=1.0,
            trend="stable",
            metadata={"dummy": True},
        )


@pytest.fixture(autouse=True)
def _clean_registry():
    """Limpia el registro antes y después de cada test."""
    # Guardar estado original
    original = dict(EngineFactory._registry)
    yield
    # Restaurar
    EngineFactory._registry = original


class TestEngineRegistration:
    """Registro de motores."""

    def test_register_and_create(self) -> None:
        """Motor registrado debe poder crearse por nombre."""
        EngineFactory.register("dummy", _DummyEngine)
        engine = EngineFactory.create("dummy")

        assert isinstance(engine, _DummyEngine)
        assert engine.name == "dummy"

    def test_register_invalid_class_raises(self) -> None:
        """Registrar clase que no es PredictionEngine debe fallar."""
        with pytest.raises(TypeError, match="PredictionEngine"):
            EngineFactory.register("bad", str)  # type: ignore[arg-type]

    def test_list_engines(self) -> None:
        """list_engines debe incluir motores registrados."""
        EngineFactory.register("dummy", _DummyEngine)
        names = EngineFactory.list_engines()

        assert "dummy" in names
        assert "baseline_moving_average" in names

    def test_unregister(self) -> None:
        """Motor desregistrado no debe poder crearse."""
        EngineFactory.register("dummy", _DummyEngine)
        EngineFactory.unregister("dummy")

        engine = EngineFactory.create("dummy")
        # Debe caer a fallback
        assert isinstance(engine, BaselineMovingAverageEngine)


class TestEngineCreation:
    """Creación de motores."""

    def test_create_baseline(self) -> None:
        """Crear baseline por nombre."""
        engine = EngineFactory.create("baseline_moving_average")
        assert isinstance(engine, BaselineMovingAverageEngine)

    def test_create_taylor(self) -> None:
        """Crear Taylor con kwargs."""
        EngineFactory.register("taylor", TaylorPredictionEngine)
        engine = EngineFactory.create("taylor", order=2, horizon=1)

        assert isinstance(engine, TaylorPredictionEngine)
        assert engine._order == 2

    def test_create_unknown_falls_to_baseline(self) -> None:
        """Motor desconocido debe retornar baseline."""
        engine = EngineFactory.create("nonexistent_engine")
        assert isinstance(engine, BaselineMovingAverageEngine)

    def test_create_with_bad_kwargs_falls_to_baseline(self) -> None:
        """Si el constructor falla, debe retornar baseline."""
        EngineFactory.register("taylor", TaylorPredictionEngine)
        # horizon=0 causa ValueError en TaylorPredictionEngine
        engine = EngineFactory.create("taylor", horizon=0)
        assert isinstance(engine, BaselineMovingAverageEngine)


class TestEngineForSensor:
    """Selección de motor por feature flags (via select_engine_for_sensor)."""

    def test_panic_button_forces_baseline(self) -> None:
        """ML_ROLLBACK_TO_BASELINE=True debe forzar baseline."""
        EngineFactory.register("taylor", TaylorPredictionEngine)

        flags = FeatureFlags(
            ML_ROLLBACK_TO_BASELINE=True,
            ML_USE_TAYLOR_PREDICTOR=True,
        )

        sel = select_engine_for_sensor(sensor_id=1, flags=flags)
        engine = EngineFactory.create(sel["engine_name"], **sel["kwargs"])
        assert isinstance(engine, BaselineMovingAverageEngine)

    def test_taylor_for_whitelisted_sensor(self) -> None:
        """Sensor en whitelist debe usar Taylor."""
        EngineFactory.register("taylor", TaylorPredictionEngine)

        flags = FeatureFlags(
            ML_USE_TAYLOR_PREDICTOR=True,
            ML_TAYLOR_SENSOR_WHITELIST="1,5,42",
        )

        sel = select_engine_for_sensor(sensor_id=5, flags=flags)
        engine = EngineFactory.create(sel["engine_name"], **sel["kwargs"])
        assert isinstance(engine, TaylorPredictionEngine)

    def test_baseline_for_non_whitelisted_sensor(self) -> None:
        """Sensor fuera de whitelist debe usar default (baseline)."""
        EngineFactory.register("taylor", TaylorPredictionEngine)

        flags = FeatureFlags(
            ML_USE_TAYLOR_PREDICTOR=True,
            ML_TAYLOR_SENSOR_WHITELIST="1,5,42",
            ML_DEFAULT_ENGINE="baseline_moving_average",
        )

        sel = select_engine_for_sensor(sensor_id=99, flags=flags)
        engine = EngineFactory.create(sel["engine_name"], **sel["kwargs"])
        assert isinstance(engine, BaselineMovingAverageEngine)

    def test_override_per_sensor(self) -> None:
        """Override por sensor tiene prioridad sobre whitelist."""
        EngineFactory.register("taylor", TaylorPredictionEngine)
        EngineFactory.register("dummy", _DummyEngine)

        flags = FeatureFlags(
            ML_USE_TAYLOR_PREDICTOR=True,
            ML_ENGINE_OVERRIDES={42: "dummy"},
        )

        sel = select_engine_for_sensor(sensor_id=42, flags=flags)
        engine = EngineFactory.create(sel["engine_name"], **sel["kwargs"])
        assert isinstance(engine, _DummyEngine)

    def test_empty_whitelist_allows_all(self) -> None:
        """Whitelist vacía permite Taylor para todos los sensores."""
        EngineFactory.register("taylor", TaylorPredictionEngine)

        flags = FeatureFlags(
            ML_USE_TAYLOR_PREDICTOR=True,
            ML_TAYLOR_SENSOR_WHITELIST=None,
        )

        sel = select_engine_for_sensor(sensor_id=999, flags=flags)
        engine = EngineFactory.create(sel["engine_name"], **sel["kwargs"])
        assert isinstance(engine, TaylorPredictionEngine)

    def test_taylor_disabled_uses_default(self) -> None:
        """Si Taylor está desactivado, usa default engine."""
        flags = FeatureFlags(
            ML_USE_TAYLOR_PREDICTOR=False,
            ML_DEFAULT_ENGINE="baseline_moving_average",
        )

        sel = select_engine_for_sensor(sensor_id=1, flags=flags)
        engine = EngineFactory.create(sel["engine_name"], **sel["kwargs"])
        assert isinstance(engine, BaselineMovingAverageEngine)

    def test_deprecated_get_engine_for_sensor_warns(self) -> None:
        """get_engine_for_sensor emits DeprecationWarning."""
        flags = FeatureFlags(ML_ROLLBACK_TO_BASELINE=True)
        with pytest.warns(DeprecationWarning, match="get_engine_for_sensor"):
            EngineFactory.get_engine_for_sensor(sensor_id=1, flags=flags)


class TestBaselineMovingAverageEngine:
    """Tests del wrapper baseline."""

    def test_predict_simple(self) -> None:
        """Predicción baseline con valores simples."""
        engine = BaselineMovingAverageEngine()
        result = engine.predict([10.0, 20.0, 30.0])

        assert result.predicted_value == pytest.approx(20.0, abs=0.01)
        assert result.trend == "stable"
        assert 0.0 <= result.confidence <= 1.0

    def test_predict_single_value(self) -> None:
        """Baseline con un solo valor."""
        engine = BaselineMovingAverageEngine()
        result = engine.predict([42.0])

        assert result.predicted_value == pytest.approx(42.0, abs=0.01)

    def test_predict_empty_raises(self) -> None:
        """Baseline con lista vacía debe lanzar ValueError."""
        engine = BaselineMovingAverageEngine()
        with pytest.raises(ValueError):
            engine.predict([])

    def test_can_handle(self) -> None:
        """Baseline puede manejar >= 1 punto."""
        engine = BaselineMovingAverageEngine()
        assert engine.can_handle(1) is True
        assert engine.can_handle(0) is False

    def test_name(self) -> None:
        """Nombre del motor."""
        engine = BaselineMovingAverageEngine()
        assert engine.name == "baseline_moving_average"
