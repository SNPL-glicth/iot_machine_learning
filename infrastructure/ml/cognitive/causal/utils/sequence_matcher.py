"""
Sequence matcher utilities for operational sequence registry.

Provides sequence matching and subsequence detection.
"""

from typing import List


class SequenceMatcher:
    """Matcher for sequence patterns."""
    
    @staticmethod
    def is_subsequence(pattern: List[int], sequence: List[int]) -> bool:
        """Check if pattern is a subsequence of sequence."""
        pattern_iter = iter(pattern)
        return all(item in pattern_iter for item in sequence)
    
    @staticmethod
    def find_matching_sequences(
        sequences: dict,
        sensor_sequence: List[int],
        min_confidence: float = 0.5,
    ) -> List:
        """Find sequences matching the given sensor sequence."""
        matching_patterns = []
        
        for pattern in sequences.values():
            if pattern.confidence < min_confidence:
                continue
            
            if SequenceMatcher.is_subsequence(pattern.sequence, sensor_sequence):
                matching_patterns.append(pattern)
        
        return matching_patterns
