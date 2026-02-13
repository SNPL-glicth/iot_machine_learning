"""Tests para EMASignalFilter y AdaptiveEMASignalFilter.

Casos:
- Constructor: validación de parámetros.
- EMA fijo: suavizado, seguimiento de señal, aislamiento por serie.
- EMA adaptativo: α sube con innovación alta, baja con señal estable.
- Batch: misma longitud, reduce ruido.
- Reset: vuelve a estado inicial.
"""

from __future__ import annotations

import math
import random

import pytest

from iot_machine_learning.infrastructure.ml.filters.ema_filter import (
    AdaptiveEMASignalFilter,
    EMASignalFilter,
)


class TestEMAConstructor:
    """Validaciones del constructor."""

    def test_alpha_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            EMASignalFilter(alpha=0.0)

    def test_alpha_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            EMASignalFilter(alpha=-0.1)

    def test_alpha_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            EMASignalFilter(alpha=1.5)

    def test_alpha_one_valid(self) -> None:
        f = EMASignalFilter(alpha=1.0)
        assert f.filter_value("1", 42.0) == 42.0


class TestEMAFiltering:
    """Filtrado EMA fijo."""

    def test_first_value_returned_raw(self) -> None:
        f = EMASignalFilter(alpha=0.3)
        assert f.filter_value("1", 20.0) == 20.0

    def test_second_value_smoothed(self) -> None:
        f = EMASignalFilter(alpha=0.3)
        f.filter_value("1", 20.0)
        result = f.filter_value("1", 30.0)
        expected = 0.3 * 30.0 + 0.7 * 20.0  # 23.0
        assert result == pytest.approx(expected)

    def test_noise_reduction(self) -> None:
        """EMA debe reducir std de señal ruidosa."""
        f = EMASignalFilter(alpha=0.2)
        random.seed(42)
        true_signal = 20.0

        raw = [true_signal + random.gauss(0, 2.0) for _ in range(100)]
        filtered = [f.filter_value("1", v) for v in raw]

        raw_std = _std(raw[10:])
        filt_std = _std(filtered[10:])
        assert filt_std < raw_std

    def test_step_change_tracking(self) -> None:
        """EMA debe seguir un cambio de nivel."""
        f = EMASignalFilter(alpha=0.3)

        for _ in range(20):
            f.filter_value("1", 20.0)
        for _ in range(30):
            result = f.filter_value("1", 30.0)

        assert result > 29.0, f"EMA no convergió: {result}"

    def test_state_isolation(self) -> None:
        """Series distintas deben tener estados independientes."""
        f = EMASignalFilter(alpha=0.3)

        f.filter_value("1", 10.0)
        f.filter_value("2", 50.0)

        s1 = f.get_state("1")
        s2 = f.get_state("2")
        assert s1 is not None and s2 is not None
        assert abs(s1.x_hat - s2.x_hat) > 30.0


class TestEMABatch:
    """Método filter() batch."""

    def test_batch_same_length(self) -> None:
        f = EMASignalFilter(alpha=0.3)
        values = [20.0 + i * 0.1 for i in range(30)]
        timestamps = [float(i) for i in range(30)]
        result = f.filter(values, timestamps)
        assert len(result) == len(values)

    def test_batch_empty(self) -> None:
        f = EMASignalFilter(alpha=0.3)
        assert f.filter([], []) == []

    def test_batch_first_value_raw(self) -> None:
        f = EMASignalFilter(alpha=0.3)
        values = [10.0, 20.0, 30.0]
        result = f.filter(values, [0.0, 1.0, 2.0])
        assert result[0] == 10.0


class TestEMAReset:
    """Reset del filtro."""

    def test_reset_single_series(self) -> None:
        f = EMASignalFilter(alpha=0.3)
        f.filter_value("1", 20.0)
        f.reset("1")
        assert f.get_state("1") is None

    def test_reset_all(self) -> None:
        f = EMASignalFilter(alpha=0.3)
        f.filter_value("1", 20.0)
        f.filter_value("2", 30.0)
        f.reset()
        assert f.get_state("1") is None
        assert f.get_state("2") is None

    def test_reset_and_refilter(self) -> None:
        f = EMASignalFilter(alpha=0.3)
        f.filter_value("1", 20.0)
        f.filter_value("1", 20.0)
        f.reset("1")
        result = f.filter_value("1", 50.0)
        assert result == 50.0  # First value after reset = raw


class TestAdaptiveEMAConstructor:
    """Validaciones del constructor adaptativo."""

    def test_alpha_min_ge_alpha_max_raises(self) -> None:
        with pytest.raises(ValueError, match="alpha_min"):
            AdaptiveEMASignalFilter(alpha_min=0.5, alpha_max=0.3)

    def test_beta_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="beta"):
            AdaptiveEMASignalFilter(beta=0.0)

    def test_scale_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="scale"):
            AdaptiveEMASignalFilter(scale=0.0)


class TestAdaptiveEMAFiltering:
    """Filtrado EMA adaptativo."""

    def test_first_value_raw(self) -> None:
        f = AdaptiveEMASignalFilter()
        assert f.filter_value("1", 20.0) == 20.0

    def test_stable_signal_low_alpha(self) -> None:
        """Señal estable → α debe bajar hacia alpha_min."""
        f = AdaptiveEMASignalFilter(alpha_min=0.05, alpha_max=0.5, scale=5.0)

        for _ in range(50):
            f.filter_value("1", 20.0)

        state = f.get_state("1")
        assert state is not None
        assert state.alpha < 0.15, f"α debería ser bajo para señal estable: {state.alpha}"

    def test_step_change_alpha_rises(self) -> None:
        """Cambio brusco → α debe subir para seguir el cambio."""
        f = AdaptiveEMASignalFilter(alpha_min=0.05, alpha_max=0.5, scale=5.0)

        for _ in range(20):
            f.filter_value("1", 20.0)

        alpha_before = f.get_state("1").alpha

        # Cambio brusco
        f.filter_value("1", 50.0)
        alpha_after = f.get_state("1").alpha

        assert alpha_after > alpha_before, (
            f"α debería subir con cambio brusco: before={alpha_before}, after={alpha_after}"
        )

    def test_noise_reduction(self) -> None:
        """Adaptive EMA debe reducir ruido."""
        f = AdaptiveEMASignalFilter(alpha_min=0.05, alpha_max=0.4, scale=3.0)
        random.seed(42)
        true_signal = 20.0

        raw = [true_signal + random.gauss(0, 2.0) for _ in range(100)]
        filtered = [f.filter_value("1", v) for v in raw]

        raw_std = _std(raw[20:])
        filt_std = _std(filtered[20:])
        assert filt_std < raw_std

    def test_batch_same_length(self) -> None:
        f = AdaptiveEMASignalFilter()
        values = [20.0 + i * 0.1 for i in range(30)]
        result = f.filter(values, [float(i) for i in range(30)])
        assert len(result) == len(values)

    def test_batch_empty(self) -> None:
        f = AdaptiveEMASignalFilter()
        assert f.filter([], []) == []


def _std(values: list[float]) -> float:
    """Helper: desviación estándar poblacional."""
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)
