"""
OperationalSequenceRegistry for persisting operational sequences.

Persists frequent sequences, operational chains, recurrent patterns, and anomaly precursors.
"""

from threading import RLock
from typing import Deque, List, Dict, Any, Optional
import time
from collections import defaultdict, deque

from domain.entities.causal import TemporalPattern
from .utils.sequence_matcher import SequenceMatcher
from .utils.sequence_statistics import SequenceStatisticsCalculator

MAX_ANOMALY_PRECURSORS = 1000
MAX_SEQUENCE_COUNTS = 5000
SEQUENCE_COUNTS_TTL_SECONDS = 86400 * 7  # 7 days


class OperationalSequenceRegistry:
    """Registry for operational sequences."""
    
    def __init__(self, max_sequences: int = 1000, max_statistics_age_seconds: float = 86400.0):
        """Initialize operational sequence registry."""
        self._max_sequences = max_sequences
        self._max_statistics_age_seconds = max_statistics_age_seconds
        self._lock = RLock()
        
        self._sequences: Dict[str, TemporalPattern] = {}
        self._sequence_counts: Dict[tuple, int] = defaultdict(int)
        self._sequence_last_updated: Dict[tuple, float] = {}
        self._anomaly_precursors: Deque[tuple] = deque(maxlen=MAX_ANOMALY_PRECURSORS)
    
    def register_sequence(self, pattern: TemporalPattern) -> None:
        """Register a temporal pattern."""
        with self._lock:
            self._sequences[pattern.pattern_id] = pattern
            
            sequence_key = tuple(pattern.sequence)
            self._sequence_counts[sequence_key] += 1
            self._sequence_last_updated[sequence_key] = time.time()
            
            self._cleanup_sequence_counts()
            
            if pattern.is_pre_anomaly:
                self._anomaly_precursors.append(sequence_key)
            
            if len(self._sequences) > self._max_sequences:
                oldest_ids = sorted(
                    self._sequences.keys(),
                    key=lambda pid: self._sequences[pid].timestamp,
                )[:len(self._sequences) - self._max_sequences]
                
                for pid in oldest_ids:
                    del self._sequences[pid]
    
    def get_sequence(self, pattern_id: str) -> Optional[TemporalPattern]:
        """Get sequence by ID."""
        with self._lock:
            return self._sequences.get(pattern_id)
    
    def get_frequent_sequences(self, min_frequency: int = 5) -> List[TemporalPattern]:
        """Get frequent sequences."""
        with self._lock:
            self._cleanup_sequence_counts()
            frequent_patterns = []
            
            for pattern in self._sequences.values():
                sequence_key = tuple(pattern.sequence)
                count = self._sequence_counts.get(sequence_key, 0)
                
                if count >= min_frequency:
                    frequent_patterns.append(pattern)
            
            return sorted(frequent_patterns, key=lambda p: p.frequency, reverse=True)
    
    def _cleanup_sequence_counts(self) -> None:
        if len(self._sequence_counts) > MAX_SEQUENCE_COUNTS:
            cutoff = time.time() - SEQUENCE_COUNTS_TTL_SECONDS
            stale_keys = [
                k for k, last_ts in self._sequence_last_updated.items()
                if last_ts < cutoff
            ]
            for k in stale_keys:
                self._sequence_counts.pop(k, None)
                self._sequence_last_updated.pop(k, None)

    def get_anomaly_precursors(self, limit: int = 50) -> List[TemporalPattern]:
        """Get anomaly precursor patterns."""
        with self._lock:
            precursor_patterns = [
                pattern for pattern in self._sequences.values()
                if pattern.is_pre_anomaly
            ]
            
            return sorted(precursor_patterns, key=lambda p: p.frequency, reverse=True)[:limit]
    
    def find_matching_sequences(
        self,
        sensor_sequence: List[int],
        min_confidence: float = 0.5,
    ) -> List[TemporalPattern]:
        """Find sequences matching the given sensor sequence."""
        with self._lock:
            return SequenceMatcher.find_matching_sequences(
                self._sequences, sensor_sequence, min_confidence
            )
    
    def get_operational_chains(self, min_length: int = 3) -> List[List[int]]:
        """Get operational chains."""
        with self._lock:
            chains = []
            
            for pattern in self._sequences.values():
                if len(pattern.sequence) >= min_length:
                    chains.append(pattern.sequence)
            
            return chains
    
    def get_sequence_statistics(self) -> Dict[str, Any]:
        """Get sequence registry statistics."""
        with self._lock:
            return SequenceStatisticsCalculator.calculate_statistics(
                self._sequences,
                self._anomaly_precursors,
                self._sequence_counts,
            )
    
    def cleanup_old_sequences(self, max_age_seconds: float = 86400 * 7) -> int:
        """Clean up old sequences."""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - max_age_seconds
            
            original_count = len(self._sequences)
            
            self._sequences = {
                pid: pattern
                for pid, pattern in self._sequences.items()
                if pattern.timestamp >= cutoff_time
            }
            
            self._cleanup_sequence_counts()
            
            return original_count - len(self._sequences)
    
    def reset(self) -> None:
        """Reset all sequences."""
        with self._lock:
            self._sequences = {}
            self._sequence_counts = defaultdict(int)
            self._sequence_last_updated = {}
            self._anomaly_precursors = deque(maxlen=MAX_ANOMALY_PRECURSORS)
