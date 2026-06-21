"""
OperationalMemoryPipeline for orchestrating memory operations.

Coordinates SemanticEventBuilder, AnomalyMemoryStore, and quality filtering.
"""

import time
from typing import Optional, Dict, Any
import logging

from .semantic_event_builder import SemanticEventBuilder
from .anomaly_memory_store import AnomalyMemoryStore
from .cognitive_memory_registry import CognitiveMemoryRegistry
from domain.entities.memory import MemoryEvent

logger = logging.getLogger(__name__)


class OperationalMemoryPipeline:
    """Pipeline for operational memory operations."""
    
    def __init__(
        self,
        event_builder: SemanticEventBuilder,
        memory_store: AnomalyMemoryStore,
        registry: CognitiveMemoryRegistry,
    ):
        """
        Initialize memory pipeline.
        
        Args:
            event_builder: Semantic event builder
            memory_store: Anomaly memory store
            registry: Cognitive memory registry
        """
        self._event_builder = event_builder
        self._memory_store = memory_store
        self._registry = registry
        
        # Deduplication cache (sensor_id -> last timestamp)
        self._dedup_cache: Dict[int, float] = {}
    
    def process_event(
        self,
        sensor_id: int,
        sensor_type: str,
        ml_features: Dict[str, Any],
        regime: str,
        anomaly_score: float,
        previous_regime: Optional[str] = None,
        transition_duration: Optional[float] = None,
    ) -> Optional[MemoryEvent]:
        """
        Process operational event and decide whether to persist.
        
        Args:
            sensor_id: Sensor identifier
            sensor_type: Sensor type
            ml_features: ML features dictionary
            regime: Current operational regime
            anomaly_score: Anomaly score
            previous_regime: Previous regime
            transition_duration: Transition duration
        
        Returns:
            MemoryEvent if persisted, None otherwise
        """
        if not self._registry.enable_memory:
            return None
        
        # 1. Evaluate event quality
        if not self._evaluate_quality(anomaly_score, ml_features):
            logger.debug(f"Event quality too low for sensor {sensor_id}")
            return None
        
        # 2. Check deduplication
        if self._is_duplicate(sensor_id, ml_features.get("timestamp", time.time())):
            logger.debug(f"Duplicate event for sensor {sensor_id}")
            return None
        
        # 3. Build semantic event
        semantic_event = self._event_builder.build(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            ml_features=ml_features,
            regime=regime,
            anomaly_score=anomaly_score,
            previous_regime=previous_regime,
            transition_duration=transition_duration,
        )
        
        # 4. Determine TTL
        ttl = self._registry.get_ttl(semantic_event.event_type)
        
        # 5. Persist asynchronously (non-blocking)
        self._persist_async(semantic_event, ttl)
        
        # 6. Update deduplication cache
        self._dedup_cache[sensor_id] = ml_features.get("timestamp", time.time())
        
        return semantic_event
    
    def _evaluate_quality(self, anomaly_score: float, ml_features: Dict[str, Any]) -> bool:
        """
        Evaluate event quality.
        
        Args:
            anomaly_score: Anomaly score
            ml_features: ML features
        
        Returns:
            True if quality is sufficient, False otherwise
        """
        # Check minimum anomaly score
        if anomaly_score < self._registry.min_anomaly_score:
            return False
        
        # Check feature variability
        dynamic_features = ml_features.get("dynamic_features", {})
        rolling_std = dynamic_features.get("rolling_std_1h", 0.0)
        if rolling_std < self._registry.min_feature_variability:
            return False
        
        return True
    
    def _is_duplicate(self, sensor_id: int, timestamp: float) -> bool:
        """
        Check if event is a duplicate.
        
        Args:
            sensor_id: Sensor identifier
            timestamp: Event timestamp
        
        Returns:
            True if duplicate, False otherwise
        """
        last_timestamp = self._dedup_cache.get(sensor_id)
        if last_timestamp is None:
            return False
        
        # Consider duplicate if within 1 second
        return abs(timestamp - last_timestamp) < 1.0
    
    def _persist_async(self, event: MemoryEvent, ttl: int) -> None:
        """
        Persist event asynchronously (non-blocking).
        
        Args:
            event: MemoryEvent to persist
            ttl: Time-to-live in seconds
        """
        # In production, use async queue or background task
        # For MVP, we persist synchronously but log as async
        try:
            self._memory_store.store(event, ttl)
        except Exception as e:
            logger.error(f"Failed to persist event: {e}")
    
    def cleanup_expired_memory(self) -> int:
        """
        Clean up expired memory.
        
        Returns:
            Number of events cleaned up
        """
        return self._memory_store.cleanup_expired()
    
    def clear_dedup_cache(self) -> None:
        """Clear deduplication cache."""
        self._dedup_cache.clear()
