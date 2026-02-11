"""Tests para TaylorPredictionAdapter y KalmanFilterAdapter.

Verifica que los adapters bridge entre Fase 1 (ml/core/) y
Enterprise (domain/ports/) funcionan correctamente.

Escenarios:
- TaylorAdapter produce Prediction (entidad de dominio) válida
- TaylorAdapter implementa PredictionPort correctamente
- KalmanAdapter filtra ventana y retorna SensorWindow
- Adapter con datos insuficientes
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.entities.prediction import Prediction
from iot_machine_learning.domain.entities.sensor_reading import (
    SensorReading,
    SensorWindow,
)
from iot_machine_learning.domain.ports.prediction_port import PredictionPort
from iot_machine_learning.infrastructure.ml.engines.taylor_adapter import (
    KalmanFilterAdapter,
    TaylorPredictionAdapter,
)


def _make_window(sensor_id: int, values: list[float]) -> SensorWindow:
    """Helper para crear SensorWindow."""
    readings = [
        SensorReading(sensor_id=sensor_id, value=v, timestamp=float(i))
        for i, v in enumerate(values)
    ]
    return SensorWindow(sensor_id=sensor_id, readings=readings)


class TestTaylorPredictionAdapter:
    """Tests para el adapter Taylor → PredictionPort."""

    def test_implements_prediction_port(self) -> None:
        """Adapter debe ser instancia de PredictionPort."""
        adapter = TaylorPredictionAdapter(order=2, horizon=1)
        assert isinstance(adapter, PredictionPort)

    def test_name_is_taylor_adapted(self) -> None:
        """Nombre del adapter debe ser 'taylor_adapted'."""
        adapter = TaylorPredictionAdapter()
        assert adapter.name == "taylor_adapted"

    def test_predict_returns_domain_prediction(self) -> None:
        """predict() debe retornar Prediction (entidad de dominio)."""
        adapter = TaylorPredictionAdapter(order=2, horizon=1)
        window = _make_window(1, [20.0 + i * 0.1 for i in range(20)])

        result = adapter.predict(window)

        assert isinstance(result, Prediction)
        assert result.series_id == "1"
        assert result.engine_name == "taylor_adapted"
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.trend in ("up", "down", "stable")

    def test_predict_linear_ramp(self) -> None:
        """Rampa lineal debe predecir continuación de tendencia."""
        adapter = TaylorPredictionAdapter(order=1, horizon=1)
        # Rampa: 20.0, 20.5, 21.0, ..., 29.5
        values = [20.0 + i * 0.5 for i in range(20)]
        window = _make_window(1, values)

        result = adapter.predict(window)

        # Debe predecir ~30.0 (siguiente paso de la rampa)
        assert 29.0 <= result.predicted_value <= 31.0
        assert result.trend == "up"

    def test_predict_stable_sensor(self) -> None:
        """Sensor estable debe predecir valor cercano al actual."""
        adapter = TaylorPredictionAdapter(order=2, horizon=1)
        values = [22.0] * 20
        window = _make_window(1, values)

        result = adapter.predict(window)

        assert abs(result.predicted_value - 22.0) < 0.5

    def test_can_handle_sufficient_points(self) -> None:
        """can_handle debe retornar True con suficientes puntos."""
        adapter = TaylorPredictionAdapter(order=2)
        assert adapter.can_handle(20) is True

    def test_can_handle_insufficient_points(self) -> None:
        """can_handle debe retornar False con pocos puntos."""
        adapter = TaylorPredictionAdapter(order=3)
        assert adapter.can_handle(1) is False

    def test_metadata_present(self) -> None:
        """Metadata de Taylor debe propagarse al Prediction."""
        adapter = TaylorPredictionAdapter(order=2, horizon=1)
        window = _make_window(1, [20.0 + i * 0.1 for i in range(20)])

        result = adapter.predict(window)

        assert result.metadata is not None
        assert isinstance(result.metadata, dict)

    def test_different_orders(self) -> None:
        """Diferentes órdenes de Taylor deben funcionar."""
        for order in (1, 2, 3):
            adapter = TaylorPredictionAdapter(order=order, horizon=1)
            window = _make_window(1, [20.0 + i * 0.1 for i in range(20)])
            result = adapter.predict(window)
            assert isinstance(result, Prediction)


class TestKalmanFilterAdapter:
    """Tests para el adapter Kalman → SensorWindow."""

    def test_filter_window_returns_sensor_window(self) -> None:
        """filter_window debe retornar SensorWindow."""
        adapter = KalmanFilterAdapter(Q=1e-5, warmup_size=5)
        window = _make_window(1, [20.0 + i * 0.1 for i in range(20)])

        filtered = adapter.filter_window(window)

        assert isinstance(filtered, SensorWindow)
        assert filtered.sensor_id == 1
        assert filtered.size == window.size

    def test_filter_reduces_noise(self) -> None:
        """Kalman debe reducir ruido en la señal."""
        import random
        random.seed(42)

        adapter = KalmanFilterAdapter(Q=1e-5, warmup_size=5)

        # Señal con ruido
        noisy_values = [20.0 + random.gauss(0, 2.0) for _ in range(50)]
        window = _make_window(1, noisy_values)

        filtered = adapter.filter_window(window)
        filtered_values = filtered.values

        # Varianza filtrada debe ser menor que la original
        mean_orig = sum(noisy_values) / len(noisy_values)
        var_orig = sum((v - mean_orig) ** 2 for v in noisy_values) / len(noisy_values)

        # Solo comparar post-warmup
        post_warmup = filtered_values[10:]
        mean_filt = sum(post_warmup) / len(post_warmup)
        var_filt = sum((v - mean_filt) ** 2 for v in post_warmup) / len(post_warmup)

        assert var_filt < var_orig, "Kalman no redujo varianza"

    def test_reset(self) -> None:
        """Reset no debe crashear."""
        adapter = KalmanFilterAdapter()
        adapter.reset(sensor_id=1)
        adapter.reset()  # Reset all
