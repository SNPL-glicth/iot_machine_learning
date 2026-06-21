"""
Dynamic feature infrastructure module for ZENIN ML cognitive pipeline.

This module provides infrastructure for dynamic feature computation including:
- RollingWindowEngine: Manages multiple rolling windows per sensor
- DynamicFeaturePipeline: Orchestrates dynamic feature computation
- FeatureMetadataRegistry: Manages feature configuration per sensor type
"""

from .pipeline import DynamicFeaturePipeline
from .rolling_window_engine import RollingWindowEngine
from .feature_metadata_registry import FeatureMetadataRegistry

__all__ = [
    "DynamicFeaturePipeline",
    "RollingWindowEngine",
    "FeatureMetadataRegistry",
]
