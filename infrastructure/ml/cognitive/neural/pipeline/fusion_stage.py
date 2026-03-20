"""Fusion stage — combines SNN and classical outputs."""

from __future__ import annotations

from typing import Dict, Tuple

from ..types import SpikePattern


class FusionStage:
    """Fuses SNN and classical outputs using weighted average.
    
    Args:
        snn_weight: Weight for SNN output [0, 1]
        classical_weight: Weight for classical output [0, 1]
    """
    
    def __init__(
        self,
        snn_weight: float = 0.5,
        classical_weight: float = 0.5,
    ) -> None:
        self.snn_weight = snn_weight
        self.classical_weight = classical_weight
    
    def process(
        self,
        snn_output: float,
        classical_output: float,
        spike_patterns: Dict[str, SpikePattern],
    ) -> Tuple[float, float]:
        """Fuse SNN and classical outputs.
        
        Args:
            snn_output: SNN output score [0, 1]
            classical_output: Classical output score [0, 1]
            spike_patterns: Output spike patterns from SNN
            
        Returns:
            Tuple of (hybrid_output, confidence)
        """
        # Weighted average
        hybrid_output = (
            self.snn_weight * snn_output +
            self.classical_weight * classical_output
        )
        
        # Compute confidence from agreement and spike activity
        confidence = self._compute_confidence(
            snn_output,
            classical_output,
            spike_patterns,
        )
        
        return hybrid_output, confidence
    
    def _compute_confidence(
        self,
        snn_output: float,
        classical_output: float,
        spike_patterns: Dict[str, SpikePattern],
    ) -> float:
        """Compute confidence from output agreement and spike activity.
        
        High confidence when SNN and classical agree.
        """
        # Agreement metric
        agreement = 1.0 - abs(snn_output - classical_output)
        
        # Base confidence from spike patterns
        spike_confidence = 0.5
        if spike_patterns:
            # Higher firing rates → higher confidence
            max_rate = max(
                (p.firing_rate for p in spike_patterns.values()),
                default=0.0
            )
            spike_confidence = min(0.9, 0.5 + max_rate * 2.0)
        
        # Combine: 60% agreement + 40% spike activity
        confidence = 0.6 * agreement + 0.4 * spike_confidence
        
        return min(0.95, max(0.3, confidence))
