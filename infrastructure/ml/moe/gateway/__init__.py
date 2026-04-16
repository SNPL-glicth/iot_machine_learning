"""Gateway module for MoE architecture.

Exports:
- MoEGateway: Main gateway implementing PredictionPort
- MoEMetadata: Execution metadata for traceability
- ContextEncoderService: Encoding logic (extracted for SRP)
- ExpertDispatcher: Expert execution (extracted for SRP)
- PredictionEnricher: Metadata enrichment (extracted for SRP)
"""

from .moe_gateway import MoEGateway
from .prediction_enricher import MoEMetadata, PredictionEnricher
from .context_encoder import ContextEncoderService
from .expert_dispatcher import ExpertDispatcher

__all__ = [
    "MoEGateway",
    "MoEMetadata",
    "ContextEncoderService",
    "ExpertDispatcher",
    "PredictionEnricher",
]
