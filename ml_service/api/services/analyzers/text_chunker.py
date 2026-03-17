"""Semantic text chunking by paragraph boundaries.

Splits a document into chunks suitable for embedding.  Each chunk
respects paragraph boundaries and stays within a token budget.

Single entry point: ``chunk_text(text, max_tokens)``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TextChunk:
    """A single chunk of document text.

    Attributes:
        index: 0-based chunk ordinal.
        text: Chunk content.
        char_start: Start offset in the original document.
        char_end: End offset (exclusive) in the original document.
    """

    index: int
    text: str
    char_start: int
    char_end: int

    @property
    def token_estimate(self) -> int:
        """Rough token count (words ≈ tokens for text2vec)."""
        return len(self.text.split())


def chunk_text(text: str, max_tokens: int = 500) -> List[TextChunk]:
    """Split *text* into semantic chunks.

    Strategy:
        1. Split on double-newline (paragraph boundaries).
        2. If a paragraph exceeds *max_tokens*, split it further on
           sentence boundaries.
        3. Merge short consecutive paragraphs into one chunk if they
           fit within the budget.

    Args:
        text: Full document text.
        max_tokens: Maximum approximate tokens per chunk.

    Returns:
        Ordered list of ``TextChunk`` objects covering the entire
        document without overlap.
    """
    if not text or not text.strip():
        return []

    paragraphs = _split_paragraphs(text)
    raw_segments: List[str] = []

    for para in paragraphs:
        word_count = len(para.split())
        if word_count <= max_tokens:
            raw_segments.append(para)
        else:
            # Paragraph too large — split on sentence boundaries
            raw_segments.extend(_split_sentences_bounded(para, max_tokens))

    # Merge small consecutive segments
    merged = _merge_small(raw_segments, max_tokens)

    # Build TextChunk list with character offsets
    chunks: List[TextChunk] = []
    search_start = 0

    for idx, segment in enumerate(merged):
        # Find the segment in the original text
        pos = text.find(segment[:80], search_start)
        if pos == -1:
            pos = search_start

        char_start = pos
        char_end = char_start + len(segment)
        search_start = char_end

        chunks.append(TextChunk(
            index=idx,
            text=segment,
            char_start=char_start,
            char_end=min(char_end, len(text)),
        ))

    return chunks


# ── Internal helpers ─────────────────────────────────────────────


def _split_paragraphs(text: str) -> List[str]:
    """Split on double-newline, preserving non-empty paragraphs."""
    parts = re.split(r'\n\s*\n', text)
    return [p.strip() for p in parts if p.strip()]


def _split_sentences_bounded(
    paragraph: str, max_tokens: int
) -> List[str]:
    """Split a long paragraph into sentence-bounded segments."""
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    segments: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        s_words = len(sentence.split())
        if current_words + s_words > max_tokens and current:
            segments.append(" ".join(current))
            current = [sentence]
            current_words = s_words
        else:
            current.append(sentence)
            current_words += s_words

    if current:
        segments.append(" ".join(current))

    return segments


def _merge_small(segments: List[str], max_tokens: int) -> List[str]:
    """Merge consecutive small segments that fit within budget."""
    if not segments:
        return []

    merged: List[str] = []
    current = segments[0]
    current_words = len(current.split())

    for seg in segments[1:]:
        seg_words = len(seg.split())
        if current_words + seg_words <= max_tokens:
            current = current + "\n\n" + seg
            current_words += seg_words
        else:
            merged.append(current)
            current = seg
            current_words = seg_words

    merged.append(current)
    return merged
