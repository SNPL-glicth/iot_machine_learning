"""Tests para DeltaSpikeClassifier.

Escenarios de producción:
- Delta spike: válvula se abre, temp sube y se mantiene.
- Noise spike: lectura outlier aislada que vuelve al nivel.
- Normal: variación dentro del rango.
- Historia insuficiente: pocos puntos pre-spike.
- Spike contra-tendencia: spike en dirección opuesta al trend.
"""

from __future__ import annotations

import random

import pytest

from iot_machine_learning.domain.entities.pattern import SpikeClassification
from iot_machine_learning.infrastructure.ml.patterns.delta_spike_classifier import (
    DeltaSpikeClassifier,
)


class TestDeltaSpikeClassification:
    """Clasificación de spikes reales."""

    def test_delta_spike_persistent_level_change(self) -> None:
        """Cambio de nivel que persiste → DELTA_SPIKE."""
        # 30 puntos a 20°C, luego sube a 30°C y se mantiene 10 puntos
        values = [20.0] * 30 + [30.0] * 10
        spike_index = 30  # Primer punto del nuevo nivel

        classifier = DeltaSpikeClassifier(
            magnitude_threshold_sigma=2.0,
            persistence_window=5,
            min_history=20,
        )
        result = classifier.classify(values, spike_index)

        assert result.classification == SpikeClassification.DELTA_SPIKE
        assert result.is_delta_spike is True
        assert result.persistence_score > 0.5
        assert result.confidence > 0.3
        assert "legítimo" in result.explanation.lower() or "magnitud" in result.explanation.lower()

    def test_noise_spike_returns_to_baseline(self) -> None:
        """Lectura outlier aislada que vuelve al nivel → NOISE_SPIKE."""
        # 30 puntos a 20°C, 1 spike a 50°C, luego vuelve a 20°C
        values = [20.0] * 30 + [50.0] + [20.0] * 10
        spike_index = 30

        classifier = DeltaSpikeClassifier(
            magnitude_threshold_sigma=2.0,
            persistence_window=5,
            min_history=20,
        )
        result = classifier.classify(values, spike_index)

        assert result.classification == SpikeClassification.NOISE_SPIKE
        assert result.is_delta_spike is False
        assert result.persistence_score < 0.5

    def test_normal_variation(self) -> None:
        """Variación pequeña dentro del rango → NORMAL."""
        random.seed(42)
        values = [20.0 + random.gauss(0, 0.1) for _ in range(40)]
        spike_index = 30  # No es realmente un spike

        classifier = DeltaSpikeClassifier(
            magnitude_threshold_sigma=2.0,
            persistence_window=5,
            min_history=20,
        )
        result = classifier.classify(values, spike_index)

        assert result.classification == SpikeClassification.NORMAL
        assert result.is_delta_spike is False

    def test_insufficient_history(self) -> None:
        """Pocos puntos pre-spike → NORMAL con confianza 0."""
        values = [20.0] * 5 + [30.0] * 5
        spike_index = 3  # Solo 3 puntos de historia

        classifier = DeltaSpikeClassifier(min_history=20)
        result = classifier.classify(values, spike_index)

        assert result.classification == SpikeClassification.NORMAL
        assert result.confidence == 0.0
        assert "insuficiente" in result.explanation.lower()

    def test_trend_aligned_spike(self) -> None:
        """Spike en dirección del trend previo → más probable delta."""
        # Tendencia ascendente + spike grande hacia arriba
        values = [20.0 + i * 0.1 for i in range(30)] + [35.0] * 10
        spike_index = 30

        classifier = DeltaSpikeClassifier(min_history=20)
        result = classifier.classify(values, spike_index)

        assert result.trend_alignment > 0.5
        # Con tendencia alineada y persistencia, debe ser delta
        assert result.classification == SpikeClassification.DELTA_SPIKE


class TestDeltaSpikeConstructor:
    """Validaciones del constructor."""

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="magnitude_threshold_sigma"):
            DeltaSpikeClassifier(magnitude_threshold_sigma=-1.0)

    def test_small_persistence_window_raises(self) -> None:
        with pytest.raises(ValueError, match="persistence_window"):
            DeltaSpikeClassifier(persistence_window=1)

    def test_small_min_history_raises(self) -> None:
        with pytest.raises(ValueError, match="min_history"):
            DeltaSpikeClassifier(min_history=2)
