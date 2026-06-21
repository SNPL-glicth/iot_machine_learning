"""
FeedbackLoopManager for managing operational feedback infrastructure.

Prepares infrastructure for operator feedback, alert usefulness, retrieval usefulness, and explainability usefulness.
"""

from typing import Dict, Any, List, Optional
import time
from dataclasses import dataclass


@dataclass
class FeedbackEntry:
    """Entry for operational feedback."""
    timestamp: float
    sensor_id: int
    feedback_type: str
    feedback_value: float
    metadata: Dict[str, Any]


class FeedbackLoopManager:
    """Manager for operational feedback infrastructure."""
    
    def __init__(self):
        """Initialize feedback loop manager."""
        self._feedback_entries: List[FeedbackEntry] = []
    
    def record_alert_feedback(
        self,
        sensor_id: int,
        usefulness: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record alert usefulness feedback.
        
        Args:
            sensor_id: Sensor identifier
            usefulness: Usefulness score [0, 1]
            metadata: Additional metadata
        """
        entry = FeedbackEntry(
            timestamp=time.time(),
            sensor_id=sensor_id,
            feedback_type="alert_usefulness",
            feedback_value=usefulness,
            metadata=metadata or {},
        )
        self._feedback_entries.append(entry)
    
    def record_retrieval_feedback(
        self,
        sensor_id: int,
        usefulness: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record retrieval usefulness feedback.
        
        Args:
            sensor_id: Sensor identifier
            usefulness: Usefulness score [0, 1]
            metadata: Additional metadata
        """
        entry = FeedbackEntry(
            timestamp=time.time(),
            sensor_id=sensor_id,
            feedback_type="retrieval_usefulness",
            feedback_value=usefulness,
            metadata=metadata or {},
        )
        self._feedback_entries.append(entry)
    
    def record_explainability_feedback(
        self,
        sensor_id: int,
        usefulness: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record explainability usefulness feedback.
        
        Args:
            sensor_id: Sensor identifier
            usefulness: Usefulness score [0, 1]
            metadata: Additional metadata
        """
        entry = FeedbackEntry(
            timestamp=time.time(),
            sensor_id=sensor_id,
            feedback_type="explainability_usefulness",
            feedback_value=usefulness,
            metadata=metadata or {},
        )
        self._feedback_entries.append(entry)
    
    def get_feedback_summary(self, feedback_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Get feedback summary.
        
        Args:
            feedback_type: Filter by feedback type (optional)
        
        Returns:
            Dictionary with feedback summary
        """
        filtered_entries = self._feedback_entries
        if feedback_type:
            filtered_entries = [e for e in self._feedback_entries if e.feedback_type == feedback_type]
        
        if not filtered_entries:
            return {
                "count": 0,
                "mean_usefulness": 0.0,
                "median_usefulness": 0.0,
                "feedback_type": feedback_type,
            }
        
        usefulness_values = [e.feedback_value for e in filtered_entries]
        
        return {
            "count": len(filtered_entries),
            "mean_usefulness": sum(usefulness_values) / len(usefulness_values),
            "median_usefulness": sorted(usefulness_values)[len(usefulness_values) // 2],
            "feedback_type": feedback_type,
        }
    
    def get_sensor_feedback(self, sensor_id: int) -> Dict[str, Any]:
        """
        Get feedback for specific sensor.
        
        Args:
            sensor_id: Sensor identifier
        
        Returns:
            Dictionary with sensor feedback summary
        """
        sensor_entries = [e for e in self._feedback_entries if e.sensor_id == sensor_id]
        
        if not sensor_entries:
            return {
                "sensor_id": sensor_id,
                "count": 0,
                "mean_usefulness": 0.0,
            }
        
        usefulness_values = [e.feedback_value for e in sensor_entries]
        
        return {
            "sensor_id": sensor_id,
            "count": len(sensor_entries),
            "mean_usefulness": sum(usefulness_values) / len(usefulness_values),
        }
    
    def cleanup_old_feedback(self, max_age_seconds: int = 86400 * 30) -> int:
        """
        Clean up old feedback entries.
        
        Args:
            max_age_seconds: Maximum age in seconds (default: 30 days)
        
        Returns:
            Number of entries cleaned up
        """
        current_time = time.time()
        cutoff_time = current_time - max_age_seconds
        
        original_count = len(self._feedback_entries)
        self._feedback_entries = [
            e for e in self._feedback_entries
            if e.timestamp >= cutoff_time
        ]
        
        return original_count - len(self._feedback_entries)
    
    def reset(self) -> None:
        """Reset all feedback entries."""
        self._feedback_entries = []
