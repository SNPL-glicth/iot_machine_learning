"""Gateway module for MoE architecture.

Exports:
- MoEGateway: Main gateway implementing PredictionPort
- MoEMetadata: Execution metadata for traceability
"""

from .moe_gateway import MoEGateway, MoEMetadata

__all__ = [
    "MoEGateway",
    "MoEMetadata",
]
