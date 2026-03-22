"""Readability metrics for text documents.

Pure function — no side effects, no I/O.
Computes sentence-level and vocabulary-level metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ReadabilityResult:
    """Result of readability analysis.

    Attributes:
        n_sentences: Number of sentences detected.
        avg_sentence_length: Average words per sentence.
        vocabulary_richness: Ratio of unique words (len>2) to total words.
        embedded_numeric_count: Count of numeric values found in text.
        sentences: Split sentence list (for downstream structural analysis).
    """

    n_sentences: int
    avg_sentence_length: float
    vocabulary_richness: float
    embedded_numeric_count: int
    sentences: List[str]


def compute_readability(text: str, word_count: int = 0) -> ReadabilityResult:
    """Compute readability metrics from text.

    Args:
        text: Raw text content.
        word_count: Pre-computed word count (0 = compute here).

    Returns:
        ``ReadabilityResult`` with all metrics.
    """
    words = text.split() if text else []
    if word_count <= 0:
        word_count = len(words)

    # Split sentences on punctuation
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    n_sentences = max(len(sentences), 1)

    avg_sentence_length = word_count / n_sentences if n_sentences else 0.0

    # Vocabulary richness: unique words (len > 2) / total words
    unique_words = set(w.lower() for w in words if len(w) > 2)
    vocabulary_richness = len(unique_words) / word_count if word_count > 0 else 0.0

    # Embedded numeric values
    numbers_in_text = re.findall(r'-?\d+\.?\d*', text)
    embedded_numeric_count = len(numbers_in_text)

    return ReadabilityResult(
        n_sentences=n_sentences,
        avg_sentence_length=round(avg_sentence_length, 1),
        vocabulary_richness=round(vocabulary_richness, 3),
        embedded_numeric_count=embedded_numeric_count,
        sentences=sentences,
    )
