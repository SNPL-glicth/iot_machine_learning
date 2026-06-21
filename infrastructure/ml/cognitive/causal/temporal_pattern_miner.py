"""
TemporalPatternMiner for mining temporal operational patterns.

Implements detection of frequent sequences, pre-anomaly patterns, transitional chains, and operational motifs.
"""

from typing import List, Dict, Any, Tuple
import time
from collections import defaultdict, Counter

from domain.entities.causal import TemporalPattern


class TemporalPatternMiner:
    """Miner for temporal operational patterns."""
    
    def __init__(
        self,
        min_support: int = 3,
        max_pattern_length: int = 5,
        max_time_window_seconds: float = 3600.0,
    ):
        """
        Initialize temporal pattern miner.
        
        Args:
            min_support: Minimum support for pattern frequency
            max_pattern_length: Maximum pattern length
            max_time_window_seconds: Maximum time window for pattern
        """
        self._min_support = min_support
        self._max_pattern_length = max_pattern_length
        self._max_time_window_seconds = max_time_window_seconds
        
        # Event sequences for pattern mining
        self._event_sequences: List[List[Tuple[int, float]]] = []
    
    def add_event_sequence(
        self,
        sensor_sequence: List[int],
        timestamps: List[float],
    ) -> None:
        """
        Add event sequence for pattern mining.
        
        Args:
            sensor_sequence: Sequence of sensor IDs
            timestamps: Timestamps for each event
        """
        if len(sensor_sequence) != len(timestamps):
            return
        
        sequence = list(zip(sensor_sequence, timestamps))
        self._event_sequences.append(sequence)
        
        # Keep only last 1000 sequences
        if len(self._event_sequences) > 1000:
            self._event_sequences = self._event_sequences[-1000:]
    
    def mine_patterns(self) -> List[TemporalPattern]:
        """
        Mine temporal patterns from event sequences.
        
        Returns:
            List of discovered temporal patterns
        """
        patterns = []
        
        # Mine frequent sequences
        frequent_sequences = self._mine_frequent_sequences()
        
        for sequence, frequency, avg_duration in frequent_sequences:
            pattern_id = f"pattern_{len(patterns)}"
            
            # Calculate confidence
            confidence = self._calculate_pattern_confidence(sequence, frequency)
            
            # Check if pattern is pre-anomaly
            is_pre_anomaly = self._is_pre_anomaly_pattern(sequence)
            
            pattern = TemporalPattern(
                pattern_id=pattern_id,
                sequence=sequence,
                frequency=frequency,
                avg_duration_seconds=avg_duration,
                confidence=confidence,
                is_pre_anomaly=is_pre_anomaly,
                timestamp=time.time(),
                pattern_metadata={
                    "support": frequency,
                    "length": len(sequence),
                },
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _mine_frequent_sequences(self) -> List[Tuple[List[int], int, float]]:
        """Mine frequent sequences using simplified algorithm."""
        sequence_counts = Counter()
        sequence_durations = defaultdict(list)
        
        for event_sequence in self._event_sequences:
            # Extract all subsequences
            for length in range(2, min(len(event_sequence), self._max_pattern_length) + 1):
                for i in range(len(event_sequence) - length + 1):
                    subsequence = event_sequence[i:i + length]
                    
                    # Check time window constraint
                    start_time = subsequence[0][1]
                    end_time = subsequence[-1][1]
                    duration = end_time - start_time
                    
                    if duration > self._max_time_window_seconds:
                        continue
                    
                    # Extract sensor sequence
                    sensor_sequence = [sensor_id for sensor_id, _ in subsequence]
                    sequence_key = tuple(sensor_sequence)
                    
                    sequence_counts[sequence_key] += 1
                    sequence_durations[sequence_key].append(duration)
        
        # Filter by minimum support
        frequent_sequences = []
        for sequence, count in sequence_counts.items():
            if count >= self._min_support:
                avg_duration = sum(sequence_durations[sequence]) / len(sequence_durations[sequence])
                frequent_sequences.append((list(sequence), count, avg_duration))
        
        return frequent_sequences
    
    def _calculate_pattern_confidence(self, sequence: List[int], frequency: int) -> float:
        """Calculate pattern confidence."""
        # Confidence based on frequency and length
        frequency_factor = min(1.0, frequency / (self._min_support * 2))
        length_factor = min(1.0, len(sequence) / self._max_pattern_length)
        
        return (frequency_factor * 0.7 + length_factor * 0.3)
    
    def _is_pre_anomaly_pattern(self, sequence: List[int]) -> bool:
        """Check if pattern is pre-anomaly."""
        # Simplified: check if pattern ends with high-frequency anomaly sensors
        # In production, this would use anomaly detection results
        return False
    
    def detect_transitional_chains(self) -> List[List[int]]:
        """
        Detect transitional chains.
        
        Returns:
            List of transitional chains
        """
        chains = []
        
        # Find common transitions
        transition_counts = Counter()
        
        for event_sequence in self._event_sequences:
            for i in range(len(event_sequence) - 1):
                transition = (event_sequence[i][0], event_sequence[i + 1][0])
                transition_counts[transition] += 1
        
        # Build chains from frequent transitions
        for (source, target), count in transition_counts.items():
            if count >= self._min_support:
                chain = [source, target]
                chains.append(chain)
        
        return chains
    
    def find_operational_motifs(self) -> List[List[int]]:
        """
        Find operational motifs (recurring patterns).
        
        Returns:
            List of operational motifs
        """
        motifs = []
        
        # Mine frequent sequences of length 3-5
        frequent_sequences = self._mine_frequent_sequences()
        
        for sequence, frequency, _ in frequent_sequences:
            if 3 <= len(sequence) <= 5:
                motifs.append(sequence)
        
        return motifs
    
    def reset(self) -> None:
        """Reset all event sequences."""
        self._event_sequences = []
