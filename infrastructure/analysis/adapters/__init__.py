"""Adaptadores simples para testing del motor unificado."""

from .common import (
    SimpleTypeDetector,
    SimpleDomainClassifier,
    SimpleFeatureExtractor,
)
from .perception import (
    SimplePerceptionCollector,
    SimpleInhibitor,
    SimpleFusion,
    SimpleSeverityClassifier,
)

__all__ = [
    "SimpleTypeDetector",
    "SimpleDomainClassifier",
    "SimpleFeatureExtractor",
    "SimplePerceptionCollector",
    "SimpleInhibitor",
    "SimpleFusion",
    "SimpleSeverityClassifier",
]
