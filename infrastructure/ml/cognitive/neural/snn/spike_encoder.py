"""Biologically realistic spike encoding with rate + temporal coding.

Encoding strategies:
- Rate coding: Firing rate encodes stimulus intensity
- Temporal coding: First-spike latency encodes stimulus intensity
- Hybrid: Combines both for richer representation
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List

from ...universal.analysis.types import InputType


class SpikeEncoder:
    """Biologically realistic spike encoder with hybrid coding.
    
    Args:
        max_rate: Maximum firing rate in Hz (biological limit ~100Hz)
        min_rate: Minimum firing rate in Hz
        use_temporal_coding: Enable first-spike latency coding
    """
    
    def __init__(
        self,
        max_rate: float = 100.0,  # Hz
        min_rate: float = 10.0,   # Hz
        use_temporal_coding: bool = True,
    ) -> None:
        self.max_rate = max_rate
        self.min_rate = min_rate
        self.use_temporal_coding = use_temporal_coding
    
    def encode(
        self,
        analysis_scores: Dict[str, float],
        input_type: InputType,
        duration_ms: float = 100.0,
    ) -> Dict[str, List[float]]:
        """Encode analysis scores to spike trains.
        
        Uses hybrid rate + temporal coding for biological realism.
        
        Args:
            analysis_scores: Dict of {analyzer_name: score [0, 1]}
            input_type: Type of input (TEXT, NUMERIC, etc.)
            duration_ms: Duration of spike train in milliseconds
            
        Returns:
            Dict of {analyzer_name: [spike_times]}
        """
        spike_trains = {}
        
        for analyzer_name, score in analysis_scores.items():
            # Normalize score to [0, 1]
            normalized_score = max(0.0, min(1.0, score))
            
            if self.use_temporal_coding:
                # Hybrid coding: rate + temporal
                spike_times = self._hybrid_code(
                    normalized_score, duration_ms
                )
            else:
                # Pure rate coding
                spike_times = self.rate_code(
                    normalized_score, duration_ms
                )
            
            spike_trains[analyzer_name] = spike_times
        
        return spike_trains
    
    def rate_code(
        self,
        score: float,
        duration_ms: float,
        max_rate: Optional[float] = None,
        dt: float = 0.1,
    ) -> List[float]:
        """Rate coding: firing rate encodes stimulus intensity.
        
        Uses Poisson process with inter-spike intervals from
        exponential distribution (biologically realistic).
        
        Args:
            score: Normalized score [0, 1]
            duration_ms: Duration in milliseconds
            max_rate: Override maximum rate (Hz)
            dt: Time step for Poisson approximation (ms)
            
        Returns:
            List of spike times in ms
        """
        max_r = max_rate if max_rate is not None else self.max_rate
        
        # Map score to firing rate
        rate_hz = self.min_rate + score * (max_r - self.min_rate)
        rate_per_ms = rate_hz / 1000.0
        
        # Generate Poisson spike train
        spike_times = []
        t = 0.0
        
        while t < duration_ms:
            # Probability of spike in interval dt
            p_spike = rate_per_ms * dt
            
            if np.random.random() < p_spike:
                spike_times.append(t)
            
            t += dt
        
        return spike_times
    
    def temporal_code(
        self,
        score: float,
        duration_ms: float,
    ) -> List[float]:
        """Temporal coding: first-spike latency encodes intensity.
        
        Higher score → earlier first spike.
        Follows biological observation in sensory neurons.
        
        Args:
            score: Normalized score [0, 1]
            duration_ms: Duration in milliseconds
            
        Returns:
            List with single spike time (or empty if score too low)
        """
        if score < 0.01:
            return []
        
        # Latency inversely proportional to score
        # High score → short latency, low score → long latency
        max_latency = duration_ms * 0.8  # Don't use full window
        latency = max_latency * (1.0 - score)
        
        # Add small jitter (biological variability)
        jitter = np.random.normal(0, 2.0)
        latency = max(0.0, min(duration_ms, latency + jitter))
        
        return [latency]
    
    def _hybrid_code(
        self,
        score: float,
        duration_ms: float,
    ) -> List[float]:
        """Hybrid rate + temporal coding.
        
        Combines first-spike latency with ongoing rate coding.
        Provides richer representation than either alone.
        
        Args:
            score: Normalized score [0, 1]
            duration_ms: Duration in milliseconds
            
        Returns:
            List of spike times combining both codes
        """
        # First spike encodes intensity via latency
        temporal_spikes = self.temporal_code(score, duration_ms)
        
        if not temporal_spikes:
            # Score too low - use minimal rate coding
            return self.rate_code(score, duration_ms, max_rate=20.0)
        
        first_spike_time = temporal_spikes[0]
        
        # After first spike, use rate coding for remaining duration
        remaining_duration = duration_ms - first_spike_time
        
        if remaining_duration > 10.0:  # Only if enough time left
            # Generate rate-coded spikes after first spike
            rate_spikes = self.rate_code(
                score, remaining_duration, max_rate=self.max_rate
            )
            
            # Shift rate spikes to start after first spike
            rate_spikes_shifted = [
                t + first_spike_time for t in rate_spikes
            ]
            
            # Combine temporal (first spike) + rate (subsequent spikes)
            all_spikes = [first_spike_time] + rate_spikes_shifted
            
            return sorted(all_spikes)
        
        return temporal_spikes
    
    def encode_constant_rate(
        self,
        score: float,
        duration_ms: float = 100.0,
    ) -> List[float]:
        """Legacy interface for constant rate encoding."""
        return self.rate_code(score, duration_ms)


from typing import Optional
