"""Text readability metrics (Spanish-aware sentence splitting).

Pure function: no I/O, no global state, no external dependencies.
Guarantees: never raises, never returns None.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import List

_SENTENCE_SPLITTER = re.compile(r"[.!?¡¿]+")
_WORD_SPLITTER = re.compile(r"\S+")
_NUMERIC_PATTERN = re.compile(r"\b\d+(?:[.,]\d+)?(?:%|°[CF])?\b")
_MIN_SENTENCE_LENGTH: int = 2
_DEFAULT_AVG_LENGTH: float = 0.0
_DEFAULT_RICHNESS: float = 1.0
_MIN_VOCABULARY_DENOMINATOR: int = 1


@dataclass(frozen=True)
class ReadabilityResult:
    """Immutable result of readability analysis.

    Attributes:
        avg_sentence_length: Mean number of words per sentence.
        n_sentences: Number of detected sentences.
        vocabulary_richness: Ratio of unique words to total words
            (type-token ratio). Higher = richer vocabulary.
        embedded_numeric_count: Number of numeric tokens found.
        sentences: List of individual sentence strings.
    """

    avg_sentence_length: float
    n_sentences: int
    vocabulary_richness: float
    embedded_numeric_count: int
    sentences: List[str]


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences handling Spanish punctuation."""
    raw = _SENTENCE_SPLITTER.split(text)
    return [s.strip() for s in raw if len(s.strip()) >= _MIN_SENTENCE_LENGTH]


def _count_words(text: str) -> int:
    """Count space-delimited tokens in the string."""
    return len(_WORD_SPLITTER.findall(text))


def compute_readability(text: str, word_count: int) -> ReadabilityResult:
    """Analyse readability metrics of *text*.

    Args:
        text: Input string to analyse.
        word_count: Pre-computed word count (may be 0 for empty text).

    Returns:
        ReadabilityResult — never None, never raises.
    """
    if not text or not text.strip() or word_count <= 0:
        return ReadabilityResult(
            avg_sentence_length=_DEFAULT_AVG_LENGTH,
            n_sentences=0,
            vocabulary_richness=_DEFAULT_RICHNESS,
            embedded_numeric_count=0,
            sentences=[],
        )

    sentences = _split_sentences(text)
    n_sentences = len(sentences)

    if n_sentences == 0:
        return ReadabilityResult(
            avg_sentence_length=float(word_count),
            n_sentences=1,
            vocabulary_richness=_compute_richness(text, word_count),
            embedded_numeric_count=len(_NUMERIC_PATTERN.findall(text)),
            sentences=[text.strip()],
        )

    avg_sentence_length = word_count / n_sentences

    richness = _compute_richness(text, word_count)
    numeric_count = len(_NUMERIC_PATTERN.findall(text))

    return ReadabilityResult(
        avg_sentence_length=round(avg_sentence_length, 2),
        n_sentences=n_sentences,
        vocabulary_richness=round(richness, 4),
        embedded_numeric_count=numeric_count,
        sentences=sentences,
    )


def _compute_richness(text: str, word_count: int) -> float:
    """Compute type-token ratio.

    Args:
        text: Input string.
        word_count: Number of words (pre-computed, > 0).

    Returns:
        Ratio of unique to total tokens, clamped to [0.0, 1.0].
    """
    tokens = [m.group().lower() for m in _WORD_SPLITTER.finditer(text)]
    if not tokens:
        return _DEFAULT_RICHNESS
    unique = len(Counter(tokens))
    denominator = max(word_count, _MIN_VOCABULARY_DENOMINATOR)
    return min(1.0, unique / denominator)
