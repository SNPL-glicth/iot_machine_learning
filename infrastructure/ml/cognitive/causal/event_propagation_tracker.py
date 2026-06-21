"""
EventPropagationTracker for tracking event propagation.

Implements tracking of temporal propagation, average propagation time, propagation frequency, and simple operational cascades.
"""

from threading import RLock
from typing import Deque, List, Dict, Any, Optional
import time
from collections import defaultdict, deque
import uuid

from domain.entities.causal import PropagationEvent
from .utils.propagation_confidence import PropagationConfidenceCalculator

MAX_COMPLETED_PROPAGATIONS = 1000
MAX_PROPAGATION_STATS_ENTRIES = 5000
PROPAGATION_STATS_TTL_SECONDS = 86400  # 24h
STALE_PROPAGATION_TTL_SECONDS = 3600   # 1h


class EventPropagationTracker:
    """Tracker for event propagation."""
    
    def __init__(self, max_propagation_window_seconds: float = 300.0):
        """Initialize propagation tracker."""
        self._max_propagation_window_seconds = max_propagation_window_seconds
        self._lock = RLock()
        
        self._active_propagations: Dict[str, Dict[str, Any]] = {}
        self._completed_propagations: Deque[PropagationEvent] = deque(maxlen=MAX_COMPLETED_PROPAGATIONS)
        self._propagation_counts: Dict[tuple, int] = defaultdict(int)
        self._propagation_times: Dict[tuple, List[float]] = defaultdict(list)
        self._propagation_last_updated: Dict[tuple, float] = {}
    
    def start_propagation(
        self,
        source_sensor_id: int,
        timestamp: float,
    ) -> str:
        """Start tracking a new propagation."""
        with self._lock:
            self._cleanup_stale_active_propagations()
            propagation_id = str(uuid.uuid4())
            
            self._active_propagations[propagation_id] = {
                "source_sensor_id": source_sensor_id,
                "start_timestamp": timestamp,
                "target_sensors": [],
                "propagation_path": [source_sensor_id],
            }
            
            return propagation_id
    
    def add_to_propagation(
        self,
        propagation_id: str,
        target_sensor_id: int,
        timestamp: float,
    ) -> None:
        """Add target sensor to propagation."""
        with self._lock:
            if propagation_id not in self._active_propagations:
                return
            
            propagation = self._active_propagations[propagation_id]
            
            if timestamp - propagation["start_timestamp"] > self._max_propagation_window_seconds:
                self.end_propagation(propagation_id, timestamp)
                return
            
            propagation["target_sensors"].append(target_sensor_id)
            propagation["propagation_path"].append(target_sensor_id)
    
    def end_propagation(
        self,
        propagation_id: str,
        end_timestamp: float,
    ) -> Optional[PropagationEvent]:
        """End propagation tracking."""
        with self._lock:
            if propagation_id not in self._active_propagations:
                return None
            
            propagation = self._active_propagations[propagation_id]
            duration = end_timestamp - propagation["start_timestamp"]
            is_cascade = len(propagation["target_sensors"]) > 2
            
            confidence = PropagationConfidenceCalculator.calculate(
                len(propagation["target_sensors"]),
                duration,
                self._max_propagation_window_seconds,
            )
            
            event = PropagationEvent(
                propagation_id=propagation_id,
                source_sensor_id=propagation["source_sensor_id"],
                target_sensors=propagation["target_sensors"],
                start_timestamp=propagation["start_timestamp"],
                end_timestamp=end_timestamp,
                propagation_duration_seconds=duration,
                propagation_path=propagation["propagation_path"],
                confidence=confidence,
                is_cascade=is_cascade,
                metadata={
                    "target_count": len(propagation["target_sensors"]),
                },
            )
            
            for target_id in propagation["target_sensors"]:
                key = (propagation["source_sensor_id"], target_id)
                self._propagation_counts[key] += 1
                self._propagation_times[key].append(duration)
                self._propagation_last_updated[key] = time.time()
            
            self._propagation_stats_cleanup()
            
            self._completed_propagations.append(event)
            del self._active_propagations[propagation_id]
            
            return event
    
    def _propagation_stats_cleanup(self) -> None:
        if len(self._propagation_counts) > MAX_PROPAGATION_STATS_ENTRIES:
            cutoff = time.time() - PROPAGATION_STATS_TTL_SECONDS
            stale_keys = [
                k for k, last_ts in self._propagation_last_updated.items()
                if last_ts < cutoff
            ]
            for k in stale_keys:
                self._propagation_counts.pop(k, None)
                self._propagation_times.pop(k, None)
                self._propagation_last_updated.pop(k, None)

    def _cleanup_stale_active_propagations(self) -> None:
        cutoff = time.time() - STALE_PROPAGATION_TTL_SECONDS
        stale_ids = [
            pid for pid, prop in self._active_propagations.items()
            if prop["start_timestamp"] < cutoff
        ]
        for pid in stale_ids:
            del self._active_propagations[pid]

    def get_propagation_statistics(
        self,
        source_sensor_id: int,
        target_sensor_id: int,
    ) -> Dict[str, float]:
        """Get propagation statistics for sensor pair."""
        with self._lock:
            self._propagation_stats_cleanup()
            key = (source_sensor_id, target_sensor_id)
            count = self._propagation_counts.get(key, 0)
            times = self._propagation_times.get(key, [])
            
            if not times:
                return {
                    "count": 0,
                    "avg_duration_seconds": 0.0,
                    "min_duration_seconds": 0.0,
                    "max_duration_seconds": 0.0,
                }
            
            return {
                "count": count,
                "avg_duration_seconds": sum(times) / len(times),
                "min_duration_seconds": min(times),
                "max_duration_seconds": max(times),
            }
    
    def get_active_propagations(self) -> List[Dict[str, Any]]:
        """Get all active propagations."""
        with self._lock:
            return [
                {
                    "propagation_id": pid,
                    **prop,
                }
                for pid, prop in self._active_propagations.items()
            ]
    
    def get_completed_propagations(
        self,
        limit: int = 100,
    ) -> List[PropagationEvent]:
        """Get completed propagations."""
        with self._lock:
            all_events = list(self._completed_propagations)
            return all_events[-limit:]
    
    def cleanup_old_propagations(self, max_age_seconds: float = 86400) -> int:
        """Clean up old completed propagations."""
        with self._lock:
            current_time = time.time()
            cutoff_time = current_time - max_age_seconds
            
            original_count = len(self._completed_propagations)
            self._completed_propagations = deque(
                [p for p in self._completed_propagations if p.end_timestamp >= cutoff_time],
                maxlen=MAX_COMPLETED_PROPAGATIONS,
            )
            
            return original_count - len(self._completed_propagations)
    
    def reset(self) -> None:
        """Reset all propagation data."""
        with self._lock:
            self._active_propagations = {}
            self._completed_propagations = deque(maxlen=MAX_COMPLETED_PROPAGATIONS)
            self._propagation_counts = defaultdict(int)
            self._propagation_times = defaultdict(list)
            self._propagation_last_updated = {}
