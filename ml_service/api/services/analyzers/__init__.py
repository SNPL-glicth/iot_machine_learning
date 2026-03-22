"""Document analysis sub-modules.

Each module has a single responsibility and exposes pure functions
or frozen dataclasses.

Top-level pipelines:
- ``text_analyzer.analyze_text_document``
- ``tabular_analyzer.analyze_tabular_document``
- ``media_analyzer.analyze_image / analyze_audio / analyze_binary``

Building blocks:
- ``keyword_config``     — centralized keyword lists + domain/action mappings
- ``text_sentiment``     — sentiment scoring
- ``text_urgency``       — urgency + severity
- ``text_readability``   — readability metrics
- ``text_structural``    — structural signal analysis
- ``text_chunker``       — semantic chunking by paragraph
- ``text_embedder``      — Weaviate text2vec vectorization
- ``text_recall``        — semantic recall of similar past documents
- ``text_pattern``       — change-point detection on sentence signals
- ``numeric_analyzer``   — per-column ML pipeline
- ``conclusion_builder`` — Explanation + semantic conclusion rendering
"""

from .text_analyzer import analyze_text_document
from iot_machine_learning.infrastructure.ml.analyzers.tabular_analyzer import analyze_tabular_document
from .media_analyzer import analyze_image, analyze_audio, analyze_binary

__all__ = [
    "analyze_text_document",
    "analyze_tabular_document",
    "analyze_image",
    "analyze_audio",
    "analyze_binary",
]
