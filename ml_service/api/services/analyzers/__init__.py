"""Document analysis sub-modules.

Each module has a single responsibility and exposes pure functions
or frozen dataclasses.

Top-level pipelines:
- ``text_analyzer.analyze_text_document``
- ``tabular_analyzer.analyze_tabular_document``
- ``media_analyzer.analyze_image / analyze_audio / analyze_binary``

Building blocks:
- ``keyword_config``   — centralized keyword lists
- ``text_sentiment``   — sentiment scoring
- ``text_urgency``     — urgency + severity
- ``text_readability`` — readability metrics
- ``text_structural``  — structural signal analysis
- ``numeric_analyzer`` — per-column ML pipeline
- ``conclusion_builder`` — Explanation + conclusion rendering
"""

from .text_analyzer import analyze_text_document
from .tabular_analyzer import analyze_tabular_document
from .media_analyzer import analyze_image, analyze_audio, analyze_binary

__all__ = [
    "analyze_text_document",
    "analyze_tabular_document",
    "analyze_image",
    "analyze_audio",
    "analyze_binary",
]
