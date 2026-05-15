"""Tests para BaselineMovingAverageEngine P3: ventana adaptativa.

Verifica que effective_window sea inversamente proporcional a noise_ratio.
"""

from __future__ import annotations

import random

import pytest

from iot_machine_learning.infrastructure.ml.engines.core.factory import (
    BaselineMovingAverageEngine,
)


class TestBaselineAdaptiveWindow:
    """P3: ventana corta con ruido alto, larga con señal estable."""

    def test_high_noise_short_window(self) -> None:
        """Ruido gaussiano extremo → effective_window debe ser pequeña."""
        random.seed(42)
        engine = BaselineMovingAverageEngine(window=20)
        # std ~100, mean ~50 → noise_ratio ~2.0 → effective = 20/2.1 ≈ 9
        values = [random.gauss(50, 100) for _ in range(100)]
        result = engine.predict(values)

        noise_ratio = result.metadata["noise_ratio"]
        effective = result.metadata["effective_window"]

        # Ruido extremo → noise_ratio > 1.0 → ventana muy corta
        assert noise_ratio > 1.0
        assert effective < 20, (
            f"Expected short window for noisy signal, got {effective}"
        )

    def test_stable_signal_long_window(self) -> None:
        """Señal estable (casi constante) → effective_window debe ser larga."""
        engine = BaselineMovingAverageEngine(window=20)
        values = [50.0 + (i * 0.01) for i in range(100)]
        result = engine.predict(values)

        noise_ratio = result.metadata["noise_ratio"]
        effective = result.metadata["effective_window"]

        # Señal muy estable → noise_ratio cercano a 0 → ventana larga
        assert noise_ratio < 0.01
        assert effective >= 20, (
            f"Expected long window for stable signal, got {effective}"
        )

    def test_metadata_has_all_fields(self) -> None:
        """Metadata debe exponer window, effective_window y noise_ratio."""
        engine = BaselineMovingAverageEngine(window=10)
        result = engine.predict([1.0, 2.0, 3.0, 4.0, 5.0])

        assert "window" in result.metadata
        assert "effective_window" in result.metadata
        assert "noise_ratio" in result.metadata

    def test_window_clamped_to_50_max(self) -> None:
        """effective_window no debe exceder 50."""
        engine = BaselineMovingAverageEngine(window=100)
        values = [50.0] * 200  # perfectamente estable
        result = engine.predict(values)

        assert result.metadata["effective_window"] <= 50

    def test_window_clamped_to_5_min(self) -> None:
        """effective_window no debe ser menor a 5."""
        engine = BaselineMovingAverageEngine(window=5)
        # Serie con ruido extremo para forzar ventana muy corta
        random.seed(1)
        values = [random.gauss(0, 100) for _ in range(100)]
        result = engine.predict(values)

        assert result.metadata["effective_window"] >= 5

    def test_single_point_noise_ratio_zero(self) -> None:
        """Con 1 punto noise_ratio debe ser 0.0 (no hay varianza)."""
        engine = BaselineMovingAverageEngine()
        result = engine.predict([42.0])

        assert result.metadata["noise_ratio"] == 0.0
