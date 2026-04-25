"""Tests para text sentiment/urgency clamping (FIX-10).
"""

from __future__ import annotations

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.universal.analysis.perception_collector import (
    UniversalPerceptionCollector,
)


class TestSentimentClamping:
    """Sentiment and urgency scores clamped to [0,1]."""

    def test_sentiment_out_of_range_clamped(self) -> None:
        """score=1.5 → normalized=1.0 (not 1.25)."""
        collector = UniversalPerceptionCollector()
        perceptions = collector._collect_text({
            "sentiment_score": 1.5,
            "sentiment_label": "extreme_positive",
        })
        sentiment = next(
            (p for p in perceptions if p.engine_name == "text_sentiment"), None
        )
        assert sentiment is not None
        assert sentiment.predicted_value == 1.0

    def test_urgency_out_of_range_clamped(self) -> None:
        """urgency=1.3 → result=1.0."""
        collector = UniversalPerceptionCollector()
        perceptions = collector._collect_text({
            "urgency_score": 1.3,
            "urgency_severity": "critical",
        })
        urgency = next(
            (p for p in perceptions if p.engine_name == "text_urgency"), None
        )
        assert urgency is not None
        assert urgency.predicted_value == 1.0

    def test_normal_range_unchanged(self) -> None:
        """score=0.5 → normalized=0.75."""
        collector = UniversalPerceptionCollector()
        perceptions = collector._collect_text({
            "sentiment_score": 0.5,
            "sentiment_label": "positive",
            "urgency_score": 0.3,
            "urgency_severity": "info",
        })
        sentiment = next(
            (p for p in perceptions if p.engine_name == "text_sentiment"), None
        )
        assert sentiment is not None
        assert sentiment.predicted_value == pytest.approx(0.75, rel=1e-3)
