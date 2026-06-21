"""
Regime detection module for ZENIN ML cognitive pipeline.

This module provides operational regime detection capabilities including:
- RegimeDetectionPipeline: Orchestrates regime detection
- OperationalRegimeClassifier: Classifies operational regimes
- RegimeStateManager: Manages regime state history
- RegimeMetadataRegistry: Manages regime configuration
- ContextualAnomalyRouter: Routes anomalies with regime context
"""

from .pipeline import RegimeDetectionPipeline
from .classifier import OperationalRegimeClassifier
from .state_manager import RegimeStateManager
from .metadata_registry import RegimeMetadataRegistry
from .router import ContextualAnomalyRouter
from .factory import RegimeDetectionFactory

__all__ = [
    "RegimeDetectionPipeline",
    "OperationalRegimeClassifier",
    "RegimeStateManager",
    "RegimeMetadataRegistry",
    "ContextualAnomalyRouter",
    "RegimeDetectionFactory",
]
