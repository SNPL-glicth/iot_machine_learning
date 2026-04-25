"""Tests for TextPatternAnalyzer (Problema 1 fix).

3 cases:
1. Escalating urgency → detects patterns and escalating trend
2. Stable text → zero patterns, stable trend
3. Insufficient sentences → returns unavailable
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.text.analyzers.text_pattern import (
    compute_text_patterns,
)


class TestTextPatternAnalyzer:
    def test_escalating_urgency_detects_patterns(self) -> None:
        sentences = [
            "El sistema funciona correctamente.",
            "Se detecto una alerta menor.",
            "Error critico en el modulo principal.",
            "Fallo emergencia total.",
        ]
        result = compute_text_patterns(sentences)
        assert result.metadata["available"] is True
        assert result.metadata["n_patterns"] > 0
        assert result.predicted_value > 0.0
        assert result.trend == "escalating"
        assert result.confidence > 0.5

    def test_stable_text_zero_patterns(self) -> None:
        sentences = [
            "El dia esta soleado.",
            "Los usuarios reportan normalidad.",
            "No hay novedades relevantes.",
            "Todo opera dentro de parametros.",
        ]
        result = compute_text_patterns(sentences)
        assert result.metadata["available"] is True
        assert result.metadata["n_patterns"] == 0
        assert result.predicted_value == 0.0
        assert result.trend == "stable"
        assert result.metadata["pattern_summary"] == "stable_narrative"

    def test_insufficient_sentences_returns_unavailable(self) -> None:
        result = compute_text_patterns(["Unica oracion."])
        assert result.metadata["available"] is False
        assert result.metadata["reason"] == "insufficient_sentences"
        assert result.predicted_value == 0.0
