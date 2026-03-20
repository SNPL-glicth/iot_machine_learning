"""Decoder stage — converts spike patterns and scores to severity classification."""

from __future__ import annotations

from typing import Dict, Tuple

from ..snn import SpikeDecoder
from ..types import SpikePattern


class DecoderStage:
    """Decodes outputs to severity classification.
    
    Args:
        spike_decoder: SpikeDecoder instance for SNN outputs
        duration_ms: Simulation duration in milliseconds
    """
    
    def __init__(
        self,
        spike_decoder: SpikeDecoder,
        duration_ms: float = 100.0,
    ) -> None:
        self.spike_decoder = spike_decoder
        self.duration_ms = duration_ms
        
        # Severity mapping
        self.severity_thresholds = {
            "info": (0.0, 0.2),
            "low": (0.2, 0.4),
            "medium": (0.4, 0.6),
            "high": (0.6, 0.8),
            "critical": (0.8, 1.0),
        }
        
        self.score_mapping = {
            "info": 0.1,
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.9,
        }
    
    def decode_snn_output(
        self,
        spike_patterns: Dict[str, SpikePattern],
    ) -> Tuple[str, float]:
        """Decode SNN spike patterns to severity.
        
        Args:
            spike_patterns: Output spike patterns from SNN
            
        Returns:
            Tuple of (severity, snn_score)
        """
        # Decode patterns
        severity, _ = self.spike_decoder.decode(
            spike_patterns,
            self.duration_ms,
        )
        
        # Map severity to score
        snn_score = self.score_mapping.get(severity, 0.5)
        
        return severity, snn_score
    
    def decode_hybrid_output(
        self,
        hybrid_output: float,
    ) -> str:
        """Decode hybrid score to severity level.
        
        Args:
            hybrid_output: Hybrid output score [0, 1]
            
        Returns:
            Severity level
        """
        for severity, (low, high) in self.severity_thresholds.items():
            if low <= hybrid_output < high:
                return severity
        
        # Default to critical if >= 0.8
        return "critical"
