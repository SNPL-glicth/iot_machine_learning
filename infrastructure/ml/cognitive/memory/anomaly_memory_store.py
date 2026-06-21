"""
AnomalyMemoryStore for persisting operational memory in Weaviate.

Basic implementation with similarity retrieval and metadata filters.
"""

import time
from typing import List, Optional, Dict, Any
import logging

from domain.entities.memory import MemoryEvent

logger = logging.getLogger(__name__)


class AnomalyMemoryStore:
    """Store for operational memory in Weaviate."""
    
    def __init__(
        self,
        weaviate_client=None,
        embedding_model: str = "text-embedding-3-small",
        batch_size: int = 100,
    ):
        """
        Initialize memory store.
        
        Args:
            weaviate_client: Weaviate client (optional, for testing)
            embedding_model: Embedding model to use
            batch_size: Batch size for operations
        """
        self._client = weaviate_client
        self._embedding_model = embedding_model
        self._batch_size = batch_size
        self._enable_storage = True
    
    def store(self, event: MemoryEvent, ttl: int) -> Optional[str]:
        """
        Store event with embedding and metadata.
        
        Args:
            event: MemoryEvent to store
            ttl: Time-to-live in seconds
        
        Returns:
            Object ID if successful, None otherwise
        """
        if not self._enable_storage or not self._client:
            logger.warning("Memory storage disabled or client not available")
            return None
        
        try:
            # Generate embedding
            embedding = self._generate_embedding(event.semantic_text)
            
            # Store in Weaviate
            object_id = self._client.data_object.create(
                class_name="OperationalMemory",
                properties={
                    "sensor_id": event.sensor_id,
                    "sensor_type": event.sensor_type,
                    "timestamp": event.timestamp,
                    "event_type": event.event_type,
                    "semantic_text": event.semantic_text,
                    "regime": event.regime,
                    "anomaly_score": event.anomaly_score,
                    "dynamic_features": event.dynamic_features,
                    "metadata": event.metadata,
                    "ttl": int(time.time()) + ttl,
                },
                vector=embedding,
            )
            
            logger.info(f"Stored event {object_id} for sensor {event.sensor_id}")
            return object_id
            
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return None
    
    def retrieve_similar(
        self,
        query_embedding: List[float],
        sensor_id: Optional[int] = None,
        regime: Optional[str] = None,
        sensor_type: Optional[str] = None,
        top_k: int = 5,
        time_window: Optional[tuple] = None,
    ) -> List[MemoryEvent]:
        """
        Retrieve similar events with filters.
        
        Args:
            query_embedding: Query vector
            sensor_id: Filter by sensor ID
            regime: Filter by regime
            sensor_type: Filter by sensor type
            top_k: Number of results
            time_window: (start, end) timestamp window
        
        Returns:
            List of similar MemoryEvents
        """
        if not self._client:
            logger.warning("Weaviate client not available")
            return []
        
        try:
            where_filter = self._build_filter(sensor_id, regime, sensor_type, time_window)
            
            results = self._client.query.get(
                class_name="OperationalMemory",
                properties=[
                    "sensor_id", "sensor_type", "timestamp", "event_type",
                    "semantic_text", "regime", "anomaly_score", "dynamic_features", "metadata"
                ],
                near_vector={"vector": query_embedding},
                where=where_filter,
                limit=top_k,
            )
            
            return [self._result_to_event(r) for r in results]
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar events: {e}")
            return []
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired memory events.
        
        Returns:
            Number of events cleaned up
        """
        if not self._client:
            return 0
        
        try:
            current_time = int(time.time())
            
            # Query expired events
            expired = self._client.query.get(
                class_name="OperationalMemory",
                where={
                    "path": ["ttl"],
                    "operator": "LessThan",
                    "valueInt": current_time,
                },
            )
            
            # Delete expired events
            count = 0
            for obj in expired:
                self._client.data_object.delete(obj["id"])
                count += 1
            
            logger.info(f"Cleaned up {count} expired events")
            return count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired events: {e}")
            return 0
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text.
        
        Args:
            text: Text to embed
        
        Returns:
            Embedding vector
        """
        # Placeholder: In production, use OpenAI or local model
        # For MVP, return dummy embedding
        # TODO: Implement actual embedding generation
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        # Convert hash to 1536-dimensional vector (OpenAI default)
        vector = []
        for i in range(1536):
            vector.append(float((hash_obj.digest()[i % 16] / 255.0)))
        return vector
    
    def _build_filter(
        self,
        sensor_id: Optional[int],
        regime: Optional[str],
        sensor_type: Optional[str],
        time_window: Optional[tuple],
    ) -> Optional[Dict[str, Any]]:
        """Build filter for Weaviate query."""
        filters = []
        
        if sensor_id is not None:
            filters.append({
                "path": ["sensor_id"],
                "operator": "Equal",
                "valueInt": sensor_id,
            })
        
        if regime is not None:
            filters.append({
                "path": ["regime"],
                "operator": "Equal",
                "valueString": regime,
            })
        
        if sensor_type is not None:
            filters.append({
                "path": ["sensor_type"],
                "operator": "Equal",
                "valueString": sensor_type,
            })
        
        if time_window is not None:
            filters.append({
                "path": ["timestamp"],
                "operator": "GreaterThan",
                "valueNumber": time_window[0],
            })
            filters.append({
                "path": ["timestamp"],
                "operator": "LessThan",
                "valueNumber": time_window[1],
            })
        
        return {"operator": "And", "operands": filters} if filters else None
    
    def _result_to_event(self, result: Dict[str, Any]) -> MemoryEvent:
        """Convert Weaviate result to MemoryEvent."""
        props = result["properties"]
        return MemoryEvent(
            sensor_id=props["sensor_id"],
            sensor_type=props["sensor_type"],
            timestamp=props["timestamp"],
            event_type=props["event_type"],
            semantic_text=props["semantic_text"],
            regime=props["regime"],
            anomaly_score=props["anomaly_score"],
            dynamic_features=props.get("dynamic_features", {}),
            metadata=props.get("metadata", {}),
        )
    
    def enable_storage(self, enabled: bool) -> None:
        """Enable or disable storage."""
        self._enable_storage = enabled
