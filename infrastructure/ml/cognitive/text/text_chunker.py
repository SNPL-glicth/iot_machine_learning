"""Text chunking for Weaviate embedding storage.

Splits text into semantically meaningful chunks (paragraphs) with
position metadata for later recall and reconstruction.

Pure function: no I/O, no global state, no external dependencies.
Guarantees: never raises, never returns None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

_MIN_CHUNK_LENGTH: int = 20
_MAX_CHUNK_LENGTH: int = 2000
_TOKENS_PER_CHAR: float = 0.25


@dataclass(frozen=True)
class TextChunk:
    """A single chunk of text with position metadata.

    Attributes:
        text: The chunk content string.
        index: Zero-based chunk index within the original document.
        char_start: Character offset of this chunk in the original text.
        char_end: Character offset after the last character of this chunk.
        token_estimate: Estimated token count (chars * 0.25).
    """

    text: str
    index: int
    char_start: int
    char_end: int
    token_estimate: int


def _split_by_paragraphs(text: str) -> List[str]:
    """Split text into non-empty paragraph segments."""
    raw = text.split("\n\n")
    return [p.strip() for p in raw if p.strip()]


def _merge_short_chunks(chunks: List[str]) -> List[str]:
    """Merge consecutive chunks shorter than MIN_CHUNK_LENGTH."""
    if not chunks:
        return []
    merged: List[str] = []
    buffer = ""
    for chunk in chunks:
        if len(buffer) < _MIN_CHUNK_LENGTH:
            buffer = (buffer + "\n\n" + chunk).strip() if buffer else chunk
        else:
            merged.append(buffer)
            buffer = chunk
    if buffer:
        merged.append(buffer)
    return merged


def _split_long_chunks(chunks: List[str]) -> List[str]:
    """Split chunks exceeding MAX_CHUNK_LENGTH at sentence boundaries."""
    result: List[str] = []
    for chunk in chunks:
        if len(chunk) <= _MAX_CHUNK_LENGTH:
            result.append(chunk)
            continue
        sentences = chunk.replace("\n", " ").split(". ")
        buffer = ""
        for sentence in sentences:
            candidate = (buffer + ". " + sentence).strip() if buffer else sentence
            if len(candidate) > _MAX_CHUNK_LENGTH and buffer:
                result.append(buffer + ".")
                buffer = sentence
            else:
                buffer = candidate
        if buffer:
            result.append(buffer + ".")
    return result


def chunk_text(full_text: str) -> List[TextChunk]:
    """Split *full_text* into a list of ``TextChunk`` objects.

    Strategy:
        1. Split by double-newline (paragraphs).
        2. Merge consecutive paragraphs shorter than ``_MIN_CHUNK_LENGTH``.
        3. Split paragraphs longer than ``_MAX_CHUNK_LENGTH`` at sentence
           boundaries.
        4. Assign position metadata (index, char offsets, token estimate).

    Args:
        full_text: The complete document text.

    Returns:
        List of ``TextChunk`` — never None, may be empty for blank text.
    """
    if not full_text or not full_text.strip():
        return []

    paragraphs = _split_by_paragraphs(full_text)
    if not paragraphs:
        return []

    paragraphs = _merge_short_chunks(paragraphs)
    paragraphs = _split_long_chunks(paragraphs)

    chunks: List[TextChunk] = []
    char_cursor = 0

    for index, paragraph in enumerate(paragraphs):
        char_start = full_text.find(paragraph, char_cursor)
        if char_start == -1:
            char_start = char_cursor
        char_end = char_start + len(paragraph)
        char_cursor = char_end

        chunks.append(
            TextChunk(
                text=paragraph,
                index=index,
                char_start=char_start,
                char_end=char_end,
                token_estimate=max(1, int(len(paragraph) * _TOKENS_PER_CHAR)),
            )
        )

    return chunks
