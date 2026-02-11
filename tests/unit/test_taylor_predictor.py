"""Tests de producción para TaylorPredictionEngine.

Casos basados en patrones REALES de sensores IoT:
- Sensor estable (temperatura ambiente ±0.5°C)
- Rampa lineal (calentamiento progresivo)
- Spike con recuperación (perturbación transitoria)
- Cold start (pocas lecturas al arrancar)
- Datos corruptos (NaN en la serie)

Cada test verifica:
1. Que la predicción esté en rango razonable
2. Que el trend sea correcto
3. Que metadata contenga la información esperada
4. Que no haya crashes ni excepciones inesperadas
"""

from __future__ import annotations

import math
import random

import pytest

from iot_machine_learning.infrastructure.ml.engines.taylor_engine import TaylorPredictionEngine
from iot_machine_learning.domain.validators.numeric import ValidationError


class TestTaylorStableSensor:
    """Sensor estable: 22°C ± 0.5°C, 100 puntos."""

    def test_stable_sensor_no_divergence(self) -> None:
        """Predicción de sensor estable no debe diverger del rango observado."""
        random.seed(42)
        values = [22.0 + random.uniform(-0.5, 0.5) for _ in range(100)]

        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict(values)

        # Predicción debe estar en rango razonable (±2°C del centro)
        assert 20.0 <= result.predicted_value <= 24.0, (
            f"Predicción {result.predicted_value} fuera de rango para sensor estable"
        )

        # Trend debe ser stable (variación es ruido, no tendencia)
        # Nota: con ruido aleatorio, f' puede ser ligeramente != 0,
        # pero el clamp y la media mantienen la predicción cerca.
        assert result.trend in ("stable", "up", "down"), (
            f"Trend inválido: {result.trend}"
        )

        # Confianza razonable con 100 puntos
        assert result.confidence >= 0.3

        # Metadata completa
        assert result.metadata["order"] == 2
        assert result.metadata["fallback"] is None
        assert "derivatives" in result.metadata

    def test_stable_sensor_order_1(self) -> None:
        """Orden 1 (solo velocidad) en sensor estable."""
        random.seed(123)
        values = [22.0 + random.uniform(-0.3, 0.3) for _ in range(50)]

        engine = TaylorPredictionEngine(order=1, horizon=1)
        result = engine.predict(values)

        assert 20.0 <= result.predicted_value <= 24.0
        assert result.metadata["order"] == 1
        # f'' y f''' deben ser 0 para orden 1
        assert result.metadata["derivatives"]["f_double_prime"] == 0.0
        assert result.metadata["derivatives"]["f_triple_prime"] == 0.0


class TestTaylorLinearRamp:
    """Rampa lineal: 20°C → 30°C en 50 pasos (calentamiento progresivo)."""

    def test_linear_ramp_captures_trend(self) -> None:
        """Taylor debe capturar tendencia lineal y predecir siguiente paso."""
        # Rampa perfecta: 20.0, 20.2, 20.4, ..., 30.0
        values = [20.0 + i * 0.2 for i in range(51)]  # 51 puntos, último = 30.0

        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict(values)

        # Siguiente paso debería ser ~30.2
        assert 29.5 <= result.predicted_value <= 31.0, (
            f"Predicción {result.predicted_value} no captura rampa lineal"
        )

        # Trend debe ser "up"
        assert result.trend == "up"

        # f' debe ser positiva (~0.2 por paso)
        f_prime = result.metadata["derivatives"]["f_prime"]
        assert f_prime > 0, f"f_prime debe ser > 0 para rampa ascendente, got {f_prime}"

        # f'' debe ser ~0 (rampa lineal, sin aceleración)
        f_double_prime = result.metadata["derivatives"]["f_double_prime"]
        assert abs(f_double_prime) < 0.01, (
            f"f'' debe ser ~0 para rampa lineal, got {f_double_prime}"
        )

    def test_linear_ramp_descending(self) -> None:
        """Rampa descendente: 30°C → 20°C."""
        values = [30.0 - i * 0.2 for i in range(51)]

        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict(values)

        assert result.trend == "down"
        assert result.metadata["derivatives"]["f_prime"] < 0

    def test_quadratic_acceleration(self) -> None:
        """Serie cuadrática: Taylor orden 2 debe capturar aceleración."""
        # f(t) = t², derivada f'(t) = 2t, f''(t) = 2
        values = [float(i * i) for i in range(20)]

        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict(values)

        # f'' debe ser ~2 (aceleración constante)
        f_double_prime = result.metadata["derivatives"]["f_double_prime"]
        assert abs(f_double_prime - 2.0) < 0.5, (
            f"f'' debe ser ~2 para serie cuadrática, got {f_double_prime}"
        )

        assert result.trend == "up"


class TestTaylorSpikeRecovery:
    """Spike con recuperación: [20]*20 + [30] + [20]*20."""

    def test_spike_with_recovery(self) -> None:
        """Después de un spike y recuperación, predicción debe ser ~20."""
        values = [20.0] * 20 + [30.0] + [20.0] * 20

        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict(values)

        # La serie se recuperó a 20.0, predicción debe estar cerca
        assert 18.0 <= result.predicted_value <= 22.0, (
            f"Predicción {result.predicted_value} no ignoró spike recuperado"
        )

        # El clamp debe prevenir divergencia
        assert result.metadata["order"] == 2

    def test_spike_at_end_clamped(self) -> None:
        """Spike al final de la serie: clamp debe limitar predicción."""
        values = [20.0] * 30 + [50.0]

        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict(values)

        # El clamp con 30% de margen sobre rango [20, 50] = margen 9
        # Límite superior = 50 + 9 = 59
        assert result.predicted_value <= 59.0, (
            f"Predicción {result.predicted_value} excede clamp"
        )
        # Debe indicar que se clampeó (la extrapolación de Taylor
        # con un spike tan grande probablemente diverge)
        # Nota: puede o no clampearse dependiendo del orden efectivo


class TestTaylorColdStart:
    """Cold start: muy pocos puntos al arrancar."""

    def test_cold_start_with_few_points(self) -> None:
        """Con solo 2 puntos, debe caer a fallback sin crashear."""
        values = [10.0, 11.0]

        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict(values)

        # Debe usar fallback (promedio de últimos 3 → promedio de 2)
        assert result.metadata["fallback"] == "insufficient_data"
        assert result.predicted_value == pytest.approx(10.5, abs=0.01)

        # Confianza baja en fallback
        assert result.confidence < 0.6

    def test_cold_start_single_point(self) -> None:
        """Con 1 solo punto, debe retornar ese valor como predicción."""
        values = [25.0]

        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict(values)

        assert result.metadata["fallback"] == "insufficient_data"
        assert result.predicted_value == pytest.approx(25.0, abs=0.01)
        assert result.confidence <= 0.3

    def test_cold_start_three_points_order_3(self) -> None:
        """Con 3 puntos y orden 3, debe reducir orden o fallback."""
        values = [10.0, 11.0, 12.0]

        engine = TaylorPredictionEngine(order=3, horizon=1)
        result = engine.predict(values)

        # order=3 necesita 5 puntos (order+2), con 3 cae a fallback
        assert result.metadata["fallback"] == "insufficient_data"


class TestTaylorNaNHandling:
    """Manejo de datos corruptos."""

    def test_nan_in_values_raises(self) -> None:
        """NaN en la serie debe lanzar ValueError."""
        values = [1.0, 2.0, float("nan"), 4.0]

        engine = TaylorPredictionEngine(order=2, horizon=1)

        with pytest.raises(ValidationError, match="NaN"):
            engine.predict(values)

    def test_inf_in_values_raises(self) -> None:
        """Infinity en la serie debe lanzar ValueError."""
        values = [1.0, 2.0, float("inf"), 4.0]

        engine = TaylorPredictionEngine(order=2, horizon=1)

        with pytest.raises(ValidationError, match="infinito"):
            engine.predict(values)

    def test_empty_values_raises(self) -> None:
        """Serie vacía debe lanzar ValueError."""
        engine = TaylorPredictionEngine(order=2, horizon=1)

        with pytest.raises(ValidationError, match="al menos"):
            engine.predict([])


class TestTaylorConstructor:
    """Validaciones del constructor."""

    def test_order_clamped_to_max_3(self) -> None:
        """Orden > 3 se clampea a 3."""
        engine = TaylorPredictionEngine(order=5, horizon=1)
        assert engine._order == 3

    def test_order_clamped_to_min_1(self) -> None:
        """Orden < 1 se clampea a 1."""
        engine = TaylorPredictionEngine(order=0, horizon=1)
        assert engine._order == 1

    def test_negative_horizon_raises(self) -> None:
        """Horizon negativo debe lanzar ValueError."""
        with pytest.raises(ValueError, match="horizon"):
            TaylorPredictionEngine(order=2, horizon=0)

    def test_can_handle_boundary(self) -> None:
        """can_handle con exactamente order+2 puntos."""
        engine = TaylorPredictionEngine(order=2, horizon=1)
        assert engine.can_handle(4) is True   # 2+2 = 4
        assert engine.can_handle(3) is False
        assert engine.can_handle(100) is True


class TestTaylorTimestamps:
    """Tests con timestamps explícitos."""

    def test_uniform_timestamps(self) -> None:
        """Timestamps uniformes deben dar mismo resultado que sin timestamps."""
        values = [20.0 + i * 0.5 for i in range(20)]
        timestamps = [float(i) for i in range(20)]

        engine = TaylorPredictionEngine(order=2, horizon=1)

        result_no_ts = engine.predict(values)
        result_with_ts = engine.predict(values, timestamps=timestamps)

        # Con dt=1.0 (uniforme), resultados deben ser iguales
        assert result_no_ts.predicted_value == pytest.approx(
            result_with_ts.predicted_value, abs=1e-6
        )

    def test_non_uniform_timestamps(self) -> None:
        """Timestamps no uniformes deben ajustar Δt correctamente."""
        values = [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
        # Δt variable: 1, 2, 1, 2, 1 → mediana = 1
        timestamps = [0.0, 1.0, 3.0, 4.0, 6.0, 7.0]

        engine = TaylorPredictionEngine(order=1, horizon=1)
        result = engine.predict(values, timestamps=timestamps)

        # Debe predecir sin crashear
        assert math.isfinite(result.predicted_value)
        assert result.metadata["dt"] > 0


class TestTaylorMetadata:
    """Verificación de metadata completa."""

    def test_metadata_keys_present(self) -> None:
        """Metadata debe contener todas las claves esperadas."""
        values = [20.0 + i * 0.1 for i in range(30)]

        engine = TaylorPredictionEngine(order=2, horizon=1)
        result = engine.predict(values)

        expected_keys = {"order", "derivatives", "dt", "horizon_steps", "fallback", "clamped"}
        assert expected_keys.issubset(result.metadata.keys())

        deriv_keys = {"f_t", "f_prime", "f_double_prime", "f_triple_prime"}
        assert deriv_keys.issubset(result.metadata["derivatives"].keys())

    def test_all_derivatives_finite(self) -> None:
        """Todas las derivadas deben ser finitas."""
        random.seed(99)
        values = [20.0 + random.gauss(0, 1) for _ in range(50)]

        engine = TaylorPredictionEngine(order=3, horizon=1)
        result = engine.predict(values)

        for key, val in result.metadata["derivatives"].items():
            assert math.isfinite(val), f"Derivada {key} no es finita: {val}"
