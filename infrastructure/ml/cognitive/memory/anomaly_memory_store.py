"""
AnomalyMemoryStore for persisting operational memory in Weaviate.

Generic implementation with configurable class name and property mappings.
"""

import time
from typing import List, Optional, Dict, Any
import logging

from domain.entities.memory import MemoryEvent

logger = logging.getLogger(__name__)


class AnomalyMemoryStore:
    """Generic store for operational memory in Weaviate.
    
    Configurable class name and property mappings for domain-agnostic usage.
    """
    
    DEFAULT_CLASS_NAME = "OperationalMemory"
    DEFAULT_PROPERTY_MAPPING = {
        "series_id": "series_id",
        "series_type": "series_type",
        "timestamp": "timestamp",
        "event_type": "event_type",
        "semantic_text": "semantic_text",
        "regime": "regime",
        "anomaly_score": "anomaly_score",
        "dynamic_features": "dynamic_features",
        "metadata": "metadata",
        "ttl": "ttl",
    }
    
    def __init__(
        self,
        weaviate_client=None,
        embedding_model: str = "text-embedding-3-small",
        batch_size: int = 100,
        class_name: str = DEFAULT_CLASS_NAME,
        property_mapping: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize memory store.
        
        Args:
            weaviate_client: Weaviate client (optional, for testing)
            embedding_model: Embedding model to use
            batch_size: Batch size for operations
            class_name: Weaviate class name for storing events
            property_mapping: Mapping from MemoryEvent fields to Weaviate properties
        """
        self._client = weaviate_client
        self._embedding_model = embedding_model
        self._batch_size = batch_size
        self._class_name = class_name
        self._property_mapping = property_mapping or self.DEFAULT_PROPERTY_MAPPING
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
            
            # Build properties dict using configurable mapping
            properties = {
                self._property_mapping["series_id"]: event.series_id,
                self._property_mapping["series_type"]: event.series_type,
                self._property_mapping["timestamp"]: event.timestamp,
                self._property_mapping["event_type"]: event.event_type,
                self._property_mapping["semantic_text"]: event.semantic_text,
                self._property_mapping["regime"]: event.regime,
                self._property_mapping["anomaly_score"]: event.anomaly_score,
                self._property_mapping["dynamic_features"]: event.dynamic_features,
                self._property_mapping["metadata"]: event.metadata,
                self._property_mapping["ttl"]: int(time.time()) + ttl,
            }
            
            # Store in Weaviate
            object_id = self._client.data_object.create(
                class_name=self._class_name,
                properties=properties,
                vector=embedding,
            )
            
            logger.info(f"Stored event {object_id} for series {event.series_id}")
            return object_id
            
        except Exception as e:
            logger.error(f"Failed to store event: {e}")
            return None
    
    def retrieve_similar(
        self,
        query_embedding: List[float],
        series_id: Optional[int] = None,
        regime: Optional[str] = None,
        series_type: Optional[str] = None,
        top_k: int = 5,
        time_window: Optional[tuple] = None,
    ) -> List[MemoryEvent]:
        """
        Retrieve similar events with filters.
        
        Args:
            query_embedding: Query vector
            series_id: Filter by series ID
            regime: Filter by regime
            series_type: Filter by series type
            top_k: Number of results
            time_window: (start, end) timestamp window
        
        Returns:
            List of similar MemoryEvents
        """
        if not self._client:
            logger.warning("Weaviate client not available")
            return []
        
        try:
            where_filter = self._build_filter(series_id, regime, series_type, time_window)
            
            # Build properties list from mapping values
            properties_list = list(self._property_mapping.values())
            
            results = self._client.query.get(
                class_name=self._class_name,
                properties=properties_list,
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
                class_name=self._class_name,
                where={
                    "path": [self._property_mapping["ttl"]],
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
        series_id: Optional[int],
        regime: Optional[str],
        series_type: Optional[str],
        time_window: Optional[tuple],
    ) -> Optional[Dict[str, Any]]:
        """Build filter for Weaviate query."""
        filters = []
        
        if series_id is not None:
            filters.append({
                "path": [self._property_mapping["series_id"]],
                "operator": "Equal",
                "valueInt": series_id,
            })
        
        if regime is not None:
            filters.append({
                "path": [self._property_mapping["regime"]],
                "operator": "Equal",
                "valueString": regime,
            })
        
        if series_type is not None:
            filters.append({
                "path": [self._property_mapping["series_type"]],
                "operator": "Equal",
                "valueString": series_type,
            })
        
        if time_window is not None:
            filters.append({
                "path": [self._property_mapping["timestamp"]],
                "operator": "GreaterThan",
                "valueNumber": time_window[0],
            })
            filters.append({
                "path": [self._property_mapping["timestamp"]],
                "operator": "LessThan",
                "valueNumber": time_window[1],
            })
        
        return {"operator": "And", "operands": filters} if filters else None
    
    def _result_to_event(self, result: Dict[str, Any]) -> MemoryEvent:
        """Convert Weaviate result to MemoryEvent."""
        props = result["properties"]
        # Reverse mapping: Weaviate property -> MemoryEvent field
        reverse_mapping = {v: k for k, v in self._property_mapping.items()}
        return MemoryEvent(
            series_id=props[reverse_mapping["series_id"]],
            series_type=props[reverse_mapping["series_type"]],
            timestamp=props[reverse_mapping["timestamp"]],
            event_type=props[reverse_mapping["event_type"]],
            semantic_text=props[reverse_mapping["semantic_text"]],
            regime=props[reverse_mapping["regime"]],
            anomaly_score=props[reverse_mapping["anomaly_score"]],
            dynamic_features=props.get(reverse_mapping["dynamic_features"], {}),
            metadata=props.get(reverse_mapping["metadata"], {}),
        )
    
    def enable_storage(self, enabled: bool) -> None:
        """Enable or disable storage."""
        self._enable_storage = enabled