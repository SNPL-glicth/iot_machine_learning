"""Tests para FilterChain (pipeline composable de filtros).

Casos:
- Cadena vacía = identity.
- Cadena de un filtro = equivalente al filtro solo.
- Cadena de múltiples filtros: Median → Kalman, EMA → Kalman.
- Batch y online.
- Reset propaga a todos los filtros.
"""

from __future__ import annotations

import math
import random

import pytest

from iot_machine_learning.infrastructure.ml.filters.ema_filter import EMASignalFilter
from iot_machine_learning.infrastructure.ml.filters.filter_chain import FilterChain
from iot_machine_learning.infrastructure.ml.filters.kalman_filter import KalmanSignalFilter
from iot_machine_learning.infrastructure.ml.filters.median_filter import MedianSignalFilter
from iot_machine_learning.infrastructure.ml.interfaces import IdentityFilter


class TestFilterChainConstruction:
    """Construcción de cadenas."""

    def test_empty_chain_is_identity(self) -> None:
        chain = FilterChain([])
        assert chain.filter_value("1", 42.0) == 42.0

    def test_none_filters_is_identity(self) -> None:
        chain = FilterChain()
        assert chain.filter_value("1", 42.0) == 42.0

    def test_n_filters(self) -> None:
        chain = FilterChain([EMASignalFilter(0.3), MedianSignalFilter(3)])
        assert chain.n_filters == 2

    def test_filter_names(self) -> None:
        chain = FilterChain([EMASignalFilter(0.3), MedianSignalFilter(3)])
        assert chain.filter_names == ["EMASignalFilter", "MedianSignalFilter"]


class TestFilterChainSingleFilter:
    """Cadena con un solo filtro = equivalente al filtro solo."""

    def test_single_ema_equivalent(self) -> None:
        ema = EMASignalFilter(alpha=0.3)
        chain = FilterChain([EMASignalFilter(alpha=0.3)])

        values = [20.0, 22.0, 18.0, 25.0, 19.0]
        ema_results = [ema.filter_value("1", v) for v in values]

        chain_results = [chain.filter_value("1", v) for v in values]

        for e, c in zip(ema_results, chain_results):
            assert e == pytest.approx(c)


class TestFilterChainComposition:
    """Cadenas de múltiples filtros."""

    def test_median_then_ema_removes_spike(self) -> None:
        """Median elimina spike, EMA suaviza el resto."""
        chain = FilterChain([
            MedianSignalFilter(window_size=5),
            EMASignalFilter(alpha=0.3),
        ])

        # Señal estable con spike
        for v in [20.0, 20.0, 20.0, 20.0]:
            chain.filter_value("1", v)

        # Spike: mediana lo elimina, EMA ve 20.0
        result = chain.filter_value("1", 100.0)
        assert result < 25.0, f"Spike no eliminado por cadena: {result}"

    def test_median_then_kalman_reduces_noise(self) -> None:
        """Median → Kalman: spike protection + noise reduction."""
        chain = FilterChain([
            MedianSignalFilter(window_size=5),
            KalmanSignalFilter(Q=1e-5, warmup_size=5),
        ])

        random.seed(42)
        true_signal = 20.0
        raw = [true_signal + random.gauss(0, 2.0) for _ in range(50)]

        filtered = [chain.filter_value("1", v) for v in raw]

        raw_std = _std(raw[15:])
        filt_std = _std(filtered[15:])
        assert filt_std < raw_std

    def test_chain_batch_same_length(self) -> None:
        chain = FilterChain([
            MedianSignalFilter(window_size=3),
            EMASignalFilter(alpha=0.3),
        ])
        values = [20.0 + i * 0.1 for i in range(30)]
        timestamps = [float(i) for i in range(30)]
        result = chain.filter(values, timestamps)
        assert len(result) == len(values)

    def test_chain_batch_empty(self) -> None:
        chain = FilterChain([EMASignalFilter(0.3)])
        assert chain.filter([], []) == []


class TestFilterChainReset:
    """Reset propaga a todos los filtros."""

    def test_reset_propagates(self) -> None:
        ema = EMASignalFilter(alpha=0.3)
        median = MedianSignalFilter(window_size=3)
        chain = FilterChain([median, ema])

        chain.filter_value("1", 20.0)
        chain.filter_value("1", 20.0)
        chain.filter_value("1", 20.0)

        chain.reset("1")

        # After reset, first value should be raw through both filters
        result = chain.filter_value("1", 50.0)
        assert result == 50.0

    def test_reset_all_propagates(self) -> None:
        chain = FilterChain([
            EMASignalFilter(alpha=0.3),
            MedianSignalFilter(window_size=3),
        ])

        chain.filter_value("1", 20.0)
        chain.filter_value("2", 30.0)
        chain.reset()

        assert chain.filter_value("1", 50.0) == 50.0
        assert chain.filter_value("2", 60.0) == 60.0


def _std(values: list[float]) -> float:
    n = len(values)
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)
