"""Encoder stage — converts input scores to spike trains."""

from __future__ import annotations

from typing import Dict, List

from ..snn import SpikeEncoder
from ...universal.analysis.types import InputType


class EncoderStage:
    """Encodes analysis scores as spike trains using rate coding.
    
    Args:
        encoder: SpikeEncoder instance
        duration_ms: Simulation duration in milliseconds
    """
    
    def __init__(
        self,
        encoder: SpikeEncoder,
        duration_ms: float = 100.0,
    ) -> None:
        self.encoder = encoder
        self.duration_ms = duration_ms
    
    def process(
        self,
        analysis_scores: Dict[str, float],
        input_type: InputType,
    ) -> Dict[str, List[float]]:
        """Encode analysis scores to spike trains.
        
        Args:
            analysis_scores: Dict of {analyzer_name: score [0, 1]}
            input_type: Type of input (TEXT, NUMERIC, etc.)
            
        Returns:
            Dict of {analyzer_name: [spike_times]}
        """
        return self.encoder.encode(
            analysis_scores=analysis_scores,
            input_type=input_type,
            duration_ms=self.duration_ms,
        )
