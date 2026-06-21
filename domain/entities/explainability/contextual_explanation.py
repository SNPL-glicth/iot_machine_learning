"""
ContextualExplanation — value object for contextual explainability with operational memory.

This is a domain entity for contextual explainability that incorporates:
- Current operational context (regime, dynamic features)
- Historical similar events from memory
- Contextual confidence calculation
- Structured recommendations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class ContextualExplanation:
    """
    Domain entity for contextual explainability (DDD value object).
    
    This represents the domain concept of contextual explanation with operational memory,
    separate from the general Explanation entity used in the cognitive pipeline.
    """
    
    sensor_id: int
    sensor_type: str
    timestamp: float
    
    # Current operational context
    current_regime: str
    anomaly_score: float
    primary_drivers: List[str]
    dynamic_context: Dict[str, Any]
    
    # Historical context
    similar_event_count: int
    historical_context: str
    historical_patterns: List[str]
    
    # Confidence and recommendations
    operational_confidence: float
    suggested_actions: List[str]
    
    # Metadata
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "timestamp": self.timestamp,
            "current_regime": self.current_regime,
            "anomaly_score": self.anomaly_score,
            "primary_drivers": self.primary_drivers,
            "dynamic_context": self.dynamic_context,
            "similar_event_count": self.similar_event_count,
            "historical_context": self.historical_context,
            "historical_patterns": self.historical_patterns,
            "operational_confidence": self.operational_confidence,
            "suggested_actions": self.suggested_actions,
            "extra": self.extra,
        }
