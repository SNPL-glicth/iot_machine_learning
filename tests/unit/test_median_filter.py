"""Tests para MedianSignalFilter.

Casos:
- Constructor: validación de parámetros.
- Filtrado: spike removal, señal constante, ventana creciente.
- Batch: misma longitud, spike removal.
- Aislamiento por serie.
- Reset.
"""

from __future__ import annotations

import math
import random

import pytest

from iot_machine_learning.infrastructure.ml.filters.median_filter import (
    MedianSignalFilter,
)


class TestMedianConstructor:
    """Validaciones del constructor."""

    def test_window_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            MedianSignalFilter(window_size=1)

    def test_window_even_raises(self) -> None:
        with pytest.raises(ValueError, match="impar"):
            MedianSignalFilter(window_size=4)

    def test_window_two_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size"):
            MedianSignalFilter(window_size=2)

    def test_window_three_valid(self) -> None:
        f = MedianSignalFilter(window_size=3)
        assert f.filter_value("1", 42.0) == 42.0


class TestMedianFiltering:
    """Filtrado de mediana."""

    def test_single_value_returned_as_is(self) -> None:
        f = MedianSignalFilter(window_size=3)
        assert f.filter_value("1", 20.0) == 20.0

    def test_spike_removed(self) -> None:
        """Un spike aislado debe ser eliminado por la mediana."""
        f = MedianSignalFilter(window_size=5)

        # Llenar buffer con valores normales
        for v in [20.0, 20.0, 20.0, 20.0]:
            f.filter_value("1", v)

        # Spike
        result = f.filter_value("1", 100.0)
        assert result == 20.0, f"Spike no eliminado: {result}"

    def test_constant_signal_unchanged(self) -> None:
        """Señal constante debe pasar sin cambios."""
        f = MedianSignalFilter(window_size=5)

        for _ in range(20):
            result = f.filter_value("1", 20.0)

        assert result == 20.0

    def test_step_change_followed(self) -> None:
        """Cambio de nivel debe ser seguido después de llenar la ventana."""
        f = MedianSignalFilter(window_size=3)

        for _ in range(5):
            f.filter_value("1", 20.0)

        # Cambio a 30
        f.filter_value("1", 30.0)
        f.filter_value("1", 30.0)
        result = f.filter_value("1", 30.0)
        assert result == 30.0

    def test_preserves_edges(self) -> None:
        """Mediana preserva bordes mejor que EMA/Kalman."""
        f = MedianSignalFilter(window_size=3)

        values = [10.0, 10.0, 10.0, 30.0, 30.0, 30.0]
        results = [f.filter_value("1", v) for v in values]

        # Después de 3 valores de 30, la mediana debe ser 30
        assert results[-1] == 30.0

    def test_state_isolation(self) -> None:
        """Series distintas deben tener buffers independientes."""
        f = MedianSignalFilter(window_size=3)

        f.filter_value("1", 10.0)
        f.filter_value("1", 10.0)
        f.filter_value("1", 10.0)

        f.filter_value("2", 50.0)
        f.filter_value("2", 50.0)
        f.filter_value("2", 50.0)

        r1 = f.filter_value("1", 10.0)
        r2 = f.filter_value("2", 50.0)

        assert r1 == 10.0
        assert r2 == 50.0


class TestMedianBatch:
    """Método filter() batch."""

    def test_batch_same_length(self) -> None:
        f = MedianSignalFilter(window_size=5)
        values = [20.0 + i * 0.1 for i in range(30)]
        timestamps = [float(i) for i in range(30)]
        result = f.filter(values, timestamps)
        assert len(result) == len(values)

    def test_batch_empty(self) -> None:
        f = MedianSignalFilter(window_size=3)
        assert f.filter([], []) == []

    def test_batch_spike_removed(self) -> None:
        """Spike en batch debe ser eliminado."""
        f = MedianSignalFilter(window_size=5)
        values = [20.0] * 10 + [100.0] + [20.0] * 10
        timestamps = [float(i) for i in range(len(values))]

        result = f.filter(values, timestamps)

        # El spike en posición 10 debe ser suavizado
        assert result[10] < 50.0, f"Spike no eliminado en batch: {result[10]}"


class TestMedianReset:
    """Reset del filtro."""

    def test_reset_single_series(self) -> None:
        f = MedianSignalFilter(window_size=3)
        f.filter_value("1", 20.0)
        f.filter_value("1", 20.0)
        f.reset("1")
        # After reset, first value should be returned as-is (buffer empty)
        result = f.filter_value("1", 50.0)
        assert result == 50.0

    def test_reset_all(self) -> None:
        f = MedianSignalFilter(window_size=3)
        f.filter_value("1", 20.0)
        f.filter_value("2", 30.0)
        f.reset()
        assert f.filter_value("1", 50.0) == 50.0
        assert f.filter_value("2", 60.0) == 60.0
