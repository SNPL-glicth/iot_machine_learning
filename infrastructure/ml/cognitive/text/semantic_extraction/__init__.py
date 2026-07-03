"""Semantic entity extraction — regex-only, sin NLP.

DESIGN NOTE: Diseño nuevo, no reconstrucción.
Ver docstring de composite_entity_extractor.py para detalles.
"""

from .composite_entity_extractor import RegexEntityExtractor
from .extractor_factory import ExtractorFactory

__all__ = [
    "RegexEntityExtractor",
    "ExtractorFactory",
]
