"""Narrative generation via numeric embedding (no hardcoded templates, no if/elif).

Architecture:
    situation_vector (18-dim) → Feedforward → embedding (8-dim, ReLU)
    → cosine similarity against phrase bank → top-k phrases
"""

from __future__ import annotations

from .embedding_network import NarrativeEmbeddingNetwork
from .phrase_bank import get_phrase_bank, PhraseEntry
from .generator import EmbeddingNarrativeGenerator

__all__ = [
    "NarrativeEmbeddingNetwork",
    "get_phrase_bank",
    "PhraseEntry",
    "EmbeddingNarrativeGenerator",
]
