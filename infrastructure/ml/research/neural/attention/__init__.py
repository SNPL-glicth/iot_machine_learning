"""Multi-Head Attention package for contextual text analysis.

Pure numpy implementation — no external ML dependencies.
"""

from __future__ import annotations

from .positional_encoding import PositionalEncoder
from .attention_embedding import LightweightEmbedding
from .multi_head_attention import MultiHeadAttention
from .attention_collector import AttentionContextCollector, AttentionContext

__all__ = [
    "PositionalEncoder",
    "LightweightEmbedding",
    "MultiHeadAttention",
    "AttentionContextCollector",
    "AttentionContext",
]
