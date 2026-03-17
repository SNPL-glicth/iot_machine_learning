"""TextSignalProfiler — maps text metrics to SignalSnapshot.

Encodes text structural features into the domain-pure ``SignalSnapshot``
so that downstream cognitive components (ExplanationBuilder, Renderer)
can work with text data using the same interface as numeric signals.

Mapping:
    n_points       ← word_count
    mean           ← avg_sentence_length
    std            ← sentence_length_std
    noise_ratio    ← vocabulary_richness (higher = richer vocabulary)
    slope          ← sentiment_score (positive = upward, negative = down)
    curvature      ← urgency_score (higher urgency = more curvature)
    regime         ← document_domain
    dt             ← 1.0 (default, not meaningful for text)
    extra          ← all text-specific metrics

No imports from ml_service — only domain layer.
Single entry point: ``TextSignalProfiler.profile()``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

from iot_machine_learning.domain.entities.explainability.signal_snapshot import (
    SignalSnapshot,
)


class TextSignalProfiler:
    """Maps text analysis metrics into a ``SignalSnapshot``.

    Stateless — safe to reuse across documents.
    """

    def profile(
        self,
        *,
        word_count: int,
        sentences: List[str],
        avg_sentence_length: float,
        vocabulary_richness: float,
        sentiment_score: float,
        urgency_score: float,
        domain: str,
        paragraph_count: int = 0,
        n_chunks: int = 0,
        embedded_numeric_count: int = 0,
        pattern_summary: str = "",
    ) -> SignalSnapshot:
        """Build a ``SignalSnapshot`` from text metrics.

        Args:
            word_count: Total words in document.
            sentences: List of sentence strings.
            avg_sentence_length: Average words per sentence.
            vocabulary_richness: Unique words / total words ratio.
            sentiment_score: Sentiment polarity [-1, 1].
            urgency_score: Urgency level [0, 1].
            domain: Classified document domain.
            paragraph_count: Number of paragraphs.
            n_chunks: Number of semantic chunks.
            embedded_numeric_count: Count of embedded numeric values.
            pattern_summary: Human-readable pattern description.

        Returns:
            ``SignalSnapshot`` with text metrics encoded.
        """
        sentence_lengths = [float(len(s.split())) for s in sentences]
        std = _std(sentence_lengths)

        extra: Dict[str, Any] = {
            "source": "text_cognitive_engine",
            "paragraph_count": paragraph_count,
            "n_chunks": n_chunks,
            "embedded_numeric_count": embedded_numeric_count,
            "vocabulary_richness": round(vocabulary_richness, 4),
            "urgency_score": round(urgency_score, 4),
            "sentiment_score": round(sentiment_score, 4),
        }
        if pattern_summary:
            extra["pattern_summary"] = pattern_summary

        return SignalSnapshot(
            n_points=word_count,
            mean=round(avg_sentence_length, 4),
            std=round(std, 4),
            noise_ratio=round(vocabulary_richness, 4),
            slope=round(sentiment_score, 4),
            curvature=round(urgency_score, 4),
            regime=domain,
            dt=1.0,
            extra=extra,
        )


def _std(values: List[float]) -> float:
    """Population standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return math.sqrt(variance)
