"""Spike decoder — converts spike patterns to severity + confidence.

Decodes output neuron spike patterns into classification results.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..types import SpikePattern


class SpikeDecoder:
    """Decodes spike patterns to severity classification and confidence.
    
    Uses population coding: severity is determined by which output neuron
    fires most frequently. Confidence is based on firing rate difference.
    
    Args:
        severity_levels: Ordered list of severity levels (low to high)
    """
    
    def __init__(
        self,
        severity_levels: List[str] = None,
    ) -> None:
        self.severity_levels = severity_levels or [
            "info", "low", "medium", "high", "critical"
        ]
    
    def decode(
        self,
        output_patterns: Dict[str, SpikePattern],
        duration_ms: float,
    ) -> Tuple[str, float]:
        """Decode output spike patterns to severity + confidence.
        
        Args:
            output_patterns: Spike patterns from output neurons
            duration_ms: Simulation duration
            
        Returns:
            Tuple of (severity, confidence)
        """
        if not output_patterns:
            return "info", 0.5
        
        # Extract firing rates
        firing_rates = {
            neuron_id: pattern.firing_rate
            for neuron_id, pattern in output_patterns.items()
        }
        
        # Winner-take-all: neuron with highest rate
        if not firing_rates:
            return "info", 0.5
        
        winner = max(firing_rates, key=firing_rates.get)
        winner_rate = firing_rates[winner]
        
        # Map neuron to severity
        severity = self._neuron_to_severity(winner)
        
        # Compute confidence based on rate separation
        confidence = self._compute_confidence(firing_rates, winner_rate)
        
        return severity, confidence
    
    def _neuron_to_severity(self, neuron_id: str) -> str:
        """Map output neuron to severity level.
        
        Args:
            neuron_id: Output neuron identifier
            
        Returns:
            Severity level
        """
        # Neuron IDs are expected to be "output_0", "output_1", etc.
        try:
            idx = int(neuron_id.split("_")[-1])
            if 0 <= idx < len(self.severity_levels):
                return self.severity_levels[idx]
        except (ValueError, IndexError):
            pass
        
        return "info"
    
    def _compute_confidence(
        self,
        firing_rates: Dict[str, float],
        winner_rate: float,
    ) -> float:
        """Compute confidence from firing rate distribution.
        
        Confidence is high when winner fires much more than others.
        
        Args:
            firing_rates: All neuron firing rates
            winner_rate: Firing rate of winning neuron
            
        Returns:
            Confidence score [0, 1]
        """
        if len(firing_rates) == 0:
            return 0.5
        
        if len(firing_rates) == 1:
            # Single neuron - moderate confidence
            return 0.7
        
        # Get runner-up rate
        rates = sorted(firing_rates.values(), reverse=True)
        winner = rates[0]
        runner_up = rates[1] if len(rates) > 1 else 0.0
        
        # Compute separation
        if winner < 1e-9:
            # No spikes - low confidence
            return 0.3
        
        separation = (winner - runner_up) / (winner + 1e-9)
        
        # Map separation to confidence
        # separation in [0, 1], higher is better
        base_confidence = 0.5
        confidence = base_confidence + 0.4 * separation
        
        return min(0.95, max(0.3, confidence))
    
    def decode_binary(
        self,
        spike_count: int,
        threshold: int = 5,
    ) -> Tuple[bool, float]:
        """Simple binary decoder (anomaly detection).
        
        Args:
            spike_count: Number of spikes
            threshold: Spike count threshold
            
        Returns:
            Tuple of (is_anomaly, confidence)
        """
        is_anomaly = spike_count >= threshold
        
        # Confidence based on distance from threshold
        distance = abs(spike_count - threshold)
        confidence = 0.5 + min(0.45, distance / (threshold + 1))
        
        return is_anomaly, confidence
