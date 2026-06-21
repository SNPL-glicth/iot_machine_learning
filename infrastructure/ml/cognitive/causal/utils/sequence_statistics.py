"""
Sequence statistics utilities for operational sequence registry.

Provides statistics calculation for sequences.
"""

from typing import Dict, Any, List


class SequenceStatisticsCalculator:
    """Calculator for sequence statistics."""
    
    @staticmethod
    def calculate_statistics(
        sequences: Dict[str, Any],
        anomaly_precursors: List,
        sequence_counts: Dict[tuple, int],
    ) -> Dict[str, Any]:
        """Calculate sequence registry statistics."""
        total_sequences = len(sequences)
        total_precursors = len(anomaly_precursors)
        
        frequencies = [pattern.frequency for pattern in sequences.values()]
        avg_frequency = sum(frequencies) / len(frequencies) if frequencies else 0.0
        
        confidences = [pattern.confidence for pattern in sequences.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total_sequences": total_sequences,
            "total_precursors": total_precursors,
            "average_frequency": avg_frequency,
            "average_confidence": avg_confidence,
            "unique_sequence_keys": len(sequence_counts),
        }
