"""Tests para FilterDiagnostic y compute_filter_diagnostic.

Casos:
- Serie vacía → diagnostic vacío.
- Longitudes distintas → diagnostic vacío.
- Señal idéntica → noise_reduction=0, distortion=0.
- Filtro que reduce ruido → noise_reduction > 0.
- Filtro que distorsiona → signal_distortion > 0.
- Lag estimation.
- Properties: is_effective, is_distorting.
- Serialización to_dict.
"""

from __future__ import annotations

import math
import random

import pytest

from iot_machine_learning.infrastructure.ml.filters.filter_diagnostic import (
    FilterDiagnostic,
    compute_filter_diagnostic,
    _estimate_lag,
)


class TestFilterDiagnosticEmpty:
    """Casos edge: vacío y longitudes distintas."""

    def test_empty_lists(self) -> None:
        d = compute_filter_diagnostic([], [])
        assert d.n_points == 0
        assert d.noise_reduction_ratio == 0.0

    def test_mismatched_lengths(self) -> None:
        d = compute_filter_diagnostic([1.0, 2.0], [1.0])
        assert d.n_points == 0

    def test_factory_empty(self) -> None:
        d = FilterDiagnostic.empty()
        assert d.n_points == 0
        assert d.raw_std == 0.0


class TestFilterDiagnosticIdentity:
    """Señal idéntica (filtro identity)."""

    def test_identical_signals(self) -> None:
        raw = [20.0, 21.0, 19.0, 22.0, 18.0]
        d = compute_filter_diagnostic(raw, list(raw))

        assert d.n_points == 5
        assert d.noise_reduction_ratio == pytest.approx(0.0)
        assert d.mean_absolute_error == pytest.approx(0.0)
        assert d.max_absolute_error == pytest.approx(0.0)
        assert d.signal_distortion == pytest.approx(0.0)
        assert d.lag_estimate == 0


class TestFilterDiagnosticNoiseReduction:
    """Filtro que reduce ruido."""

    def test_noise_reduction_positive(self) -> None:
        """Señal filtrada con menor std → noise_reduction > 0."""
        random.seed(42)
        raw = [20.0 + random.gauss(0, 2.0) for _ in range(50)]
        # Simular filtrado: promediar con vecinos
        filtered = [raw[0]] + [
            (raw[i - 1] + raw[i] + raw[i + 1]) / 3.0
            for i in range(1, len(raw) - 1)
        ] + [raw[-1]]

        d = compute_filter_diagnostic(raw, filtered)

        assert d.noise_reduction_ratio > 0.0, (
            f"Noise reduction debería ser positivo: {d.noise_reduction_ratio}"
        )
        assert d.filtered_std < d.raw_std

    def test_is_effective(self) -> None:
        """Filtro que reduce ruido sin distorsionar → is_effective."""
        random.seed(42)
        raw = [20.0 + random.gauss(0, 3.0) for _ in range(100)]
        filtered = [raw[0]]
        for i in range(1, len(raw)):
            filtered.append(0.3 * raw[i] + 0.7 * filtered[-1])

        d = compute_filter_diagnostic(raw, filtered)
        assert d.is_effective


class TestFilterDiagnosticDistortion:
    """Filtro que distorsiona la señal."""

    def test_level_shift_distortion(self) -> None:
        """Filtro que desplaza el nivel → signal_distortion > 0."""
        raw = [20.0] * 50
        filtered = [25.0] * 50  # Desplazamiento de +5

        d = compute_filter_diagnostic(raw, filtered)
        assert d.signal_distortion > 0.0
        assert d.is_distorting

    def test_constant_signal_no_distortion(self) -> None:
        raw = [20.0] * 20
        filtered = [20.0] * 20
        d = compute_filter_diagnostic(raw, filtered)
        assert d.signal_distortion == pytest.approx(0.0)
        assert not d.is_distorting


class TestFilterDiagnosticLag:
    """Estimación de lag."""

    def test_no_lag_identical(self) -> None:
        raw = [float(i) for i in range(20)]
        d = compute_filter_diagnostic(raw, list(raw))
        assert d.lag_estimate == 0

    def test_lag_estimation_shifted(self) -> None:
        """Señal desplazada k posiciones → lag ≈ k."""
        # Use a long ramp so that shifting by 2 is clearly detectable
        # raw[k:] should correlate best with filtered[:n-k] at k=2
        n = 60
        raw = [math.sin(i * 0.3) + 10.0 for i in range(n)]
        # filtered is raw delayed by 2: filtered[i] = raw[i-2]
        filtered = [raw[0], raw[0]] + raw[:-2]

        lag = _estimate_lag(raw, filtered, max_lag=5)
        assert lag >= 1, f"Expected lag >= 1 for shifted signal, got {lag}"

    def test_short_series_zero_lag(self) -> None:
        lag = _estimate_lag([1.0, 2.0], [1.0, 2.0])
        assert lag == 0


class TestFilterDiagnosticSerialization:
    """Serialización."""

    def test_to_dict_keys(self) -> None:
        d = FilterDiagnostic(
            n_points=10, raw_std=2.0, filtered_std=1.0,
            noise_reduction_ratio=0.5, mean_absolute_error=0.3,
            max_absolute_error=1.2, lag_estimate=1, signal_distortion=0.1,
        )
        result = d.to_dict()

        expected_keys = {
            "n_points", "raw_std", "filtered_std", "noise_reduction_ratio",
            "mean_absolute_error", "max_absolute_error", "lag_estimate",
            "signal_distortion",
        }
        assert set(result.keys()) == expected_keys

    def test_to_dict_values_rounded(self) -> None:
        d = FilterDiagnostic(raw_std=1.123456789)
        result = d.to_dict()
        assert result["raw_std"] == round(1.123456789, 8)
