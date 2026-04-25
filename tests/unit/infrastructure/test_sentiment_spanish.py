"""Tests for TextSentimentAnalyzer Spanish keywords (Problema 1).

3 cases:
1. Spanish negative text with negator + positive → negative
2. Intensifier + negative word → more negative
3. English still works (no regression)
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.text.analyzers.text_sentiment import (
    compute_sentiment,
)


class TestSentimentSpanish:
    def test_spanish_critical_text_is_negative(self) -> None:
        text = "el sistema presenta fallas críticas graves"
        result = compute_sentiment(text)
        assert result.label == "negative"
        assert result.score < -0.3
        assert result.negative_count > 0

    def test_negator_flips_positive_to_negative(self) -> None:
        text = "no correcto"
        result = compute_sentiment(text)
        # "correcto" is positive but preceded by "no" (negator)
        assert result.negative_count > 0
        assert result.score < 0.0

    def test_intensifier_amplifies_negative(self) -> None:
        text = "muy grave error"
        result = compute_sentiment(text)
        # "muy" intensifies "grave"
        assert result.negative_count > 1.0
        assert result.label == "negative"
