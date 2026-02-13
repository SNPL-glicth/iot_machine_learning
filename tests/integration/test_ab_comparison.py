"""Integration tests para A/B testing: baseline vs Taylor.

Simula escenarios completos de producción donde ambos motores predicen
en paralelo y se comparan métricas (MAE, RMSE).

Escenarios:
1. Sensor estable: ambos motores deben ser similares
2. Sensor con tendencia: Taylor debe ganar
3. Sensor con ruido alto: baseline puede ganar (Taylor sensible a ruido)
4. Flujo completo con ABTester: record → compute → summary
5. Feature flags controlan qué motor se usa
"""

from __future__ import annotations

import math
import random

import pytest

from iot_machine_learning.application.use_cases.select_engine import (
    select_engine_for_sensor,
)
from iot_machine_learning.infrastructure.ml.engines.engine_factory import (
    BaselineMovingAverageEngine,
    EngineFactory,
)
from iot_machine_learning.infrastructure.ml.engines.taylor_engine import TaylorPredictionEngine
from iot_machine_learning.ml_service.config.feature_flags import FeatureFlags
from iot_machine_learning.ml_service.metrics.ab_testing import (
    ABTester,
    ABTestResult,
    reset_ab_tester,
)


@pytest.fixture(autouse=True)
def _clean_ab_tester():
    """Limpia el ABTester antes y después de cada test."""
    reset_ab_tester()
    yield
    reset_ab_tester()


@pytest.fixture
def _register_taylor():
    """Registra Taylor en la factory."""
    original = dict(EngineFactory._registry)
    EngineFactory.register("taylor", TaylorPredictionEngine)
    yield
    EngineFactory._registry = original


class TestABTesterFlow:
    """Flujo completo del ABTester."""

    def test_record_and_compute(self) -> None:
        """Registrar predicciones y calcular resultados."""
        tester = ABTester()

        # Simular 20 predicciones
        random.seed(42)
        for _ in range(20):
            actual = 20.0 + random.gauss(0, 1)
            baseline_pred = actual + random.gauss(0, 0.5)
            taylor_pred = actual + random.gauss(0, 0.3)  # Taylor más preciso

            tester.record_prediction(
                sensor_id=1,
                actual_value=actual,
                baseline_pred=baseline_pred,
                taylor_pred=taylor_pred,
            )

        result = tester.compute_results(sensor_id=1)

        assert result is not None
        assert result.sensor_id == 1
        assert result.n_samples == 20
        assert result.baseline_mae > 0
        assert result.taylor_mae > 0
        assert result.confidence > 0

    def test_insufficient_samples_returns_none(self) -> None:
        """Menos de 10 muestras retorna None."""
        tester = ABTester()

        for i in range(5):
            tester.record_prediction(
                sensor_id=1,
                actual_value=20.0,
                baseline_pred=20.1,
                taylor_pred=19.9,
            )

        result = tester.compute_results(sensor_id=1)
        assert result is None

    def test_nan_values_ignored(self) -> None:
        """Valores NaN/Inf no deben registrarse."""
        tester = ABTester()

        tester.record_prediction(
            sensor_id=1,
            actual_value=float("nan"),
            baseline_pred=20.0,
            taylor_pred=20.0,
        )

        tester.record_prediction(
            sensor_id=1,
            actual_value=20.0,
            baseline_pred=float("inf"),
            taylor_pred=20.0,
        )

        # No debe haber datos registrados
        result = tester.compute_results(sensor_id=1)
        assert result is None

    def test_summary_no_data(self) -> None:
        """Summary sin datos debe retornar status no_data."""
        tester = ABTester()
        summary = tester.get_summary()
        assert summary["status"] == "no_data"

    def test_summary_with_data(self) -> None:
        """Summary con datos debe incluir estadísticas."""
        tester = ABTester()

        random.seed(42)
        for sensor_id in (1, 2, 3):
            for _ in range(15):
                actual = 20.0 + random.gauss(0, 1)
                tester.record_prediction(
                    sensor_id=sensor_id,
                    actual_value=actual,
                    baseline_pred=actual + random.gauss(0, 0.5),
                    taylor_pred=actual + random.gauss(0, 0.3),
                )

        tester.compute_all_results()
        summary = tester.get_summary()

        assert summary["status"] == "active"
        assert summary["total_sensors"] == 3
        assert "taylor_wins" in summary
        assert "baseline_wins" in summary
        assert "ties" in summary
        assert "avg_improvement_pct" in summary

    def test_clear_sensor(self) -> None:
        """Limpiar datos de un sensor específico."""
        tester = ABTester()

        for _ in range(15):
            tester.record_prediction(1, 20.0, 20.1, 19.9)
            tester.record_prediction(2, 30.0, 30.1, 29.9)

        tester.compute_results(1)
        tester.compute_results(2)

        tester.clear_sensor(1)

        assert tester.get_sensor_result(1) is None
        assert tester.get_sensor_result(2) is not None


class TestABStableSensor:
    """A/B con sensor estable: baseline suele ganar (promedia ruido)."""

    def test_stable_sensor_baseline_advantage(self) -> None:
        """En sensor estable con ruido, baseline suele ser mejor o empatar.

        Esto es ESPERADO: Taylor usa derivadas de los últimos puntos,
        que en una señal estable son puro ruido.  Baseline promedia
        todo el ruido, produciendo una estimación más suave.

        El test verifica que:
        1. Ambos motores producen resultados válidos
        2. Las MAE de ambos son razonables (< 1.0 para ruido ±0.3)
        3. El sistema no crashea
        """
        baseline = BaselineMovingAverageEngine()
        taylor = TaylorPredictionEngine(order=2, horizon=1)
        tester = ABTester()

        random.seed(42)
        history: list[float] = []
        for i in range(60):
            value = 22.0 + random.gauss(0, 0.3)
            history.append(value)

            if len(history) >= 10:
                window = history[-30:]

                baseline_result = baseline.predict(window)
                taylor_result = taylor.predict(window)

                tester.record_prediction(
                    sensor_id=1,
                    actual_value=value,
                    baseline_pred=baseline_result.predicted_value,
                    taylor_pred=taylor_result.predicted_value,
                )

        result = tester.compute_results(sensor_id=1)
        assert result is not None

        # Ambas MAE deben ser razonables para ruido ±0.3
        assert result.baseline_mae < 1.0, (
            f"Baseline MAE demasiado alta: {result.baseline_mae}"
        )
        assert result.taylor_mae < 1.0, (
            f"Taylor MAE demasiado alta: {result.taylor_mae}"
        )

        # En sensor estable, baseline suele ganar o empatar
        assert result.winner in ("baseline", "tie"), (
            f"En sensor estable, se espera baseline o tie, got {result.winner}"
        )


class TestABTrendingSensor:
    """A/B con sensor con tendencia: Taylor debe capturar mejor."""

    def test_trending_sensor_taylor_advantage(self) -> None:
        """En sensor con tendencia lineal, Taylor debe tener menor MAE."""
        baseline = BaselineMovingAverageEngine()
        taylor = TaylorPredictionEngine(order=1, horizon=1)
        tester = ABTester()

        # Serie con tendencia lineal clara + ruido mínimo
        random.seed(42)
        history: list[float] = []
        for i in range(80):
            value = 20.0 + i * 0.1 + random.gauss(0, 0.05)
            history.append(value)

            if len(history) >= 10:
                window = history[-30:]

                baseline_result = baseline.predict(window)
                taylor_result = taylor.predict(window)

                # El "actual" es el siguiente valor de la tendencia
                actual_next = 20.0 + (i + 1) * 0.1

                tester.record_prediction(
                    sensor_id=1,
                    actual_value=actual_next,
                    baseline_pred=baseline_result.predicted_value,
                    taylor_pred=taylor_result.predicted_value,
                )

        result = tester.compute_results(sensor_id=1)
        assert result is not None

        # Taylor debe tener menor MAE que baseline para tendencia lineal
        assert result.taylor_mae < result.baseline_mae, (
            f"Taylor MAE ({result.taylor_mae:.4f}) debe ser < "
            f"Baseline MAE ({result.baseline_mae:.4f}) para tendencia lineal"
        )
        assert result.winner == "taylor"


class TestABWithFeatureFlags:
    """A/B testing controlado por feature flags."""

    def test_ab_disabled_by_default(self) -> None:
        """A/B testing desactivado por defecto."""
        flags = FeatureFlags()
        assert flags.ML_ENABLE_AB_TESTING is False

    def test_ab_enabled_via_flags(self) -> None:
        """A/B testing se activa vía feature flags."""
        flags = FeatureFlags(ML_ENABLE_AB_TESTING=True)
        assert flags.ML_ENABLE_AB_TESTING is True

    def test_panic_button_overrides_ab(self, _register_taylor: None) -> None:
        """Panic button debe forzar baseline incluso con A/B activo."""
        flags = FeatureFlags(
            ML_ROLLBACK_TO_BASELINE=True,
            ML_USE_TAYLOR_PREDICTOR=True,
            ML_ENABLE_AB_TESTING=True,
        )

        selection = select_engine_for_sensor(sensor_id=1, flags=flags)
        engine = EngineFactory.create(selection["engine_name"], **selection["kwargs"])
        assert isinstance(engine, BaselineMovingAverageEngine)

    def test_whitelist_controls_ab_scope(self, _register_taylor: None) -> None:
        """Solo sensores en whitelist participan en A/B con Taylor."""
        flags = FeatureFlags(
            ML_USE_TAYLOR_PREDICTOR=True,
            ML_TAYLOR_SENSOR_WHITELIST="1,5",
        )

        sel_in = select_engine_for_sensor(sensor_id=1, flags=flags)
        sel_out = select_engine_for_sensor(sensor_id=99, flags=flags)
        engine_in = EngineFactory.create(sel_in["engine_name"], **sel_in["kwargs"])
        engine_out = EngineFactory.create(sel_out["engine_name"], **sel_out["kwargs"])

        assert isinstance(engine_in, TaylorPredictionEngine)
        assert isinstance(engine_out, BaselineMovingAverageEngine)


class TestABTestResultProperties:
    """Propiedades del ABTestResult."""

    def test_improvement_positive_means_taylor_better(self) -> None:
        """improvement_pct positivo = Taylor mejor que baseline."""
        tester = ABTester()

        for _ in range(20):
            tester.record_prediction(
                sensor_id=1,
                actual_value=20.0,
                baseline_pred=21.0,  # Error = 1.0
                taylor_pred=20.5,    # Error = 0.5
            )

        result = tester.compute_results(sensor_id=1)
        assert result is not None
        assert result.improvement_pct > 0
        assert result.winner == "taylor"

    def test_improvement_negative_means_baseline_better(self) -> None:
        """improvement_pct negativo = baseline mejor que Taylor."""
        tester = ABTester()

        for _ in range(20):
            tester.record_prediction(
                sensor_id=1,
                actual_value=20.0,
                baseline_pred=20.5,  # Error = 0.5
                taylor_pred=21.0,    # Error = 1.0
            )

        result = tester.compute_results(sensor_id=1)
        assert result is not None
        assert result.improvement_pct < 0
        assert result.winner == "baseline"

    def test_tie_within_margin(self) -> None:
        """Errores similares (< 5% diferencia) deben ser tie."""
        tester = ABTester()

        for _ in range(20):
            tester.record_prediction(
                sensor_id=1,
                actual_value=20.0,
                baseline_pred=20.5,  # Error = 0.5
                taylor_pred=20.48,   # Error = 0.48 (4% mejor, < 5% margen)
            )

        result = tester.compute_results(sensor_id=1)
        assert result is not None
        assert result.winner == "tie"
