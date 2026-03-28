"""Hybrid text embeddings package.

Provides character, word, and phrase-level embeddings with
adaptive domain learning and entropy-based filtering.
"""

from .char_encoder import CharacterEncoder
from .entity_detector import EntityResult, HybridEntityDetector
from .entropy_filter import EntropyFilter
from .hybrid_embedder import HybridEmbedder
from .phrase_encoder import PhraseEncoder
from .word_encoder import WordEncoder

__all__ = [
    "CharacterEncoder",
    "EntropyFilter",
    "WordEncoder",
    "PhraseEncoder",
    "HybridEmbedder",
    "HybridEntityDetector",
    "EntityResult",
]
