"""Fusion module for MoE architecture.

Exports:
- SparseFusionLayer: Weighted fusion of k experts
- FusionWeights: Normalized weights metadata
"""

from .sparse_fusion import SparseFusionLayer, FusionWeights

__all__ = [
    "SparseFusionLayer",
    "FusionWeights",
]