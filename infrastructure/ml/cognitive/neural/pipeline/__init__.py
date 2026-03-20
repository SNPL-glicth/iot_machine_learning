"""Neural pipeline stages — modular hybrid engine components.

Each stage is a pure function or stateless class that processes data
through the neural analysis pipeline.

Stages:
    encoder_stage: Input scores → spike trains
    snn_stage: Spike trains → SNN output + patterns
    classical_stage: Input scores → classical output
    fusion_stage: SNN + classical → hybrid output
    decoder_stage: Hybrid output → severity + confidence
"""

from .encoder_stage import EncoderStage
from .snn_stage import SNNStage
from .classical_stage import ClassicalStage
from .fusion_stage import FusionStage
from .decoder_stage import DecoderStage

__all__ = [
    "EncoderStage",
    "SNNStage",
    "ClassicalStage",
    "FusionStage",
    "DecoderStage",
]
