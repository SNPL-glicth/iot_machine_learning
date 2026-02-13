"""Tests para Kalman con Q adaptativo.

Casos:
- Q adaptativo sigue cambios de régimen más rápido que Q fijo.
- Q sube cuando hay innovaciones grandes (step change).
- Q baja cuando la señal es estable.
- Backward compatible: adaptive_Q=False = comportamiento original.
- Constructor: validación de innovation_window.
"""

from __future__ import annotations

import math
import random

import pytest

from iot_machine_learning.infrastructure.ml.filters.kalman_filter import KalmanSignalFilter
from iot_machine_learning.infrastructure.ml.filters.kalman_math import (
    KalmanState,
    adaptive_kalman_update,
    _compute_adaptive_Q,
)


class TestAdaptiveQConstructor:
    """Validaciones del constructor con adaptive_Q."""

    def test_innovation_window_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="innovation_window"):
            KalmanSignalFilter(Q=1e-5, adaptive_Q=True, innovation_window=2)

    def test_default_innovation_window(self) -> None:
        kf = KalmanSignalFilter(Q=1e-5, adaptive_Q=True)
        assert kf._innovation_window == 20

    def test_adaptive_false_is_default(self) -> None:
        kf = KalmanSignalFilter(Q=1e-5)
        assert kf._adaptive_Q is False


class TestAdaptiveQMath:
    """Funciones matemáticas puras de Q adaptativo."""

    def test_compute_adaptive_Q_stable(self) -> None:
        """Innovaciones pequeñas → Q bajo."""
        innovations = [0.01, -0.01, 0.02, -0.02, 0.01]
        q = _compute_adaptive_Q(innovations)
        assert q < 0.001

    def test_compute_adaptive_Q_volatile(self) -> None:
        """Innovaciones grandes → Q alto."""
        innovations = [5.0, -5.0, 4.0, -4.0, 6.0]
        q = _compute_adaptive_Q(innovations)
        assert q > 0.1

    def test_adaptive_update_modifies_Q(self) -> None:
        """adaptive_kalman_update debe modificar Q in-place."""
        state = KalmanState(
            x_hat=20.0, P=1.0, Q=1e-5, R=1.0,
            initialized=True, _innovation_window_size=10,
        )
        initial_Q = state.Q

        # Feed large innovations to force Q adaptation
        for v in [30.0, 10.0, 35.0, 5.0]:
            adaptive_kalman_update(state, v)

        assert state.Q != initial_Q, "Q should have been adapted"
        assert state.Q > initial_Q, "Q should increase with large innovations"

    def test_adaptive_update_innovations_stored(self) -> None:
        """Innovaciones deben acumularse en el estado."""
        state = KalmanState(
            x_hat=20.0, P=1.0, Q=1e-5, R=1.0,
            initialized=True, _innovation_window_size=10,
        )

        adaptive_kalman_update(state, 25.0)
        adaptive_kalman_update(state, 22.0)

        assert len(state._innovations) == 2

    def test_innovation_window_trimmed(self) -> None:
        """Ventana de innovaciones no debe exceder el tamaño configurado."""
        state = KalmanState(
            x_hat=20.0, P=1.0, Q=1e-5, R=1.0,
            initialized=True, _innovation_window_size=5,
        )

        for i in range(20):
            adaptive_kalman_update(state, 20.0 + i * 0.1)

        assert len(state._innovations) <= 5


class TestAdaptiveQFiltering:
    """Filtrado con Q adaptativo end-to-end."""

    def test_backward_compatible_fixed_Q(self) -> None:
        """adaptive_Q=False debe dar resultados idénticos al original."""
        kf_fixed = KalmanSignalFilter(Q=1e-5, warmup_size=5, adaptive_Q=False)
        kf_orig = KalmanSignalFilter(Q=1e-5, warmup_size=5)

        random.seed(42)
        values = [20.0 + random.gauss(0, 1.0) for _ in range(30)]

        results_fixed = [kf_fixed.filter_value("1", v) for v in values]
        results_orig = [kf_orig.filter_value("1", v) for v in values]

        for f, o in zip(results_fixed, results_orig):
            assert f == pytest.approx(o)

    def test_adaptive_follows_step_change_faster(self) -> None:
        """Adaptive Q debe converger más rápido a un nuevo nivel.

        Usamos solo 5 pasos post-cambio para capturar la diferencia
        antes de que ambos converjan.
        """
        kf_fixed = KalmanSignalFilter(Q=1e-5, warmup_size=5, adaptive_Q=False)
        kf_adaptive = KalmanSignalFilter(Q=1e-5, warmup_size=5, adaptive_Q=True)

        # Warmup + estable a 20
        for _ in range(20):
            kf_fixed.filter_value("1", 20.0)
            kf_adaptive.filter_value("1", 20.0)

        # Step change a 30 — solo 5 pasos para ver la diferencia
        for _ in range(5):
            fixed_val = kf_fixed.filter_value("1", 30.0)
            adaptive_val = kf_adaptive.filter_value("1", 30.0)

        # Adaptive debe estar más cerca de 30 (converge más rápido)
        fixed_error = abs(30.0 - fixed_val)
        adaptive_error = abs(30.0 - adaptive_val)
        assert adaptive_error < fixed_error, (
            f"Adaptive error ({adaptive_error:.6f}) debería ser menor que "
            f"fixed error ({fixed_error:.6f})"
        )

    def test_adaptive_noise_reduction(self) -> None:
        """Adaptive Q debe seguir reduciendo ruido en señal estable."""
        kf = KalmanSignalFilter(Q=1e-5, warmup_size=10, adaptive_Q=True)

        random.seed(42)
        true_signal = 20.0
        raw = [true_signal + random.gauss(0, 2.0) for _ in range(100)]
        filtered = [kf.filter_value("1", v) for v in raw]

        raw_std = _std(raw[15:])
        filt_std = _std(filtered[15:])
        assert filt_std < raw_std

    def test_adaptive_batch_same_length(self) -> None:
        kf = KalmanSignalFilter(Q=1e-5, warmup_size=5, adaptive_Q=True)
        values = [20.0 + i * 0.1 for i in range(30)]
        timestamps = [float(i) for i in range(30)]
        result = kf.filter(values, timestamps)
        assert len(result) == len(values)

    def test_adaptive_batch_warmup_raw(self) -> None:
        """Primeras warmup_size lecturas en batch deben ser crudas."""
        kf = KalmanSignalFilter(Q=1e-5, warmup_size=5, adaptive_Q=True)
        values = [20.0 + i * 0.1 for i in range(20)]
        timestamps = [float(i) for i in range(20)]
        filtered = kf.filter(values, timestamps)

        for i in range(5):
            assert filtered[i] == values[i]

    def test_adaptive_Q_state_has_innovation_window(self) -> None:
        """Estado debe tener innovation_window_size configurado."""
        kf = KalmanSignalFilter(
            Q=1e-5, warmup_size=5, adaptive_Q=True, innovation_window=15
        )

        for v in [20.0, 20.1, 20.2, 20.3, 20.4]:
            kf.filter_value("1", v)

        state = kf.get_state("1")
        assert state is not None
        assert state._innovation_window_size == 15


def _std(values: list[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)
