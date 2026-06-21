"""
HistoricalSimilarityRetriever for retrieving similar historical events.

Implements top-k retrieval with regime and sensor filters.
"""

import time
from typing import List, Optional, Dict, Any
import logging

from .anomaly_memory_store import AnomalyMemoryStore
from domain.entities.memory import MemoryEvent

logger = logging.getLogger(__name__)


class HistoricalSimilarityRetriever:
    """Retriever for similar historical events."""
    
    def __init__(
        self,
        memory_store: AnomalyMemoryStore,
        min_similarity_threshold: float = 0.7,
    ):
        """
        Initialize similarity retriever.
        
        Args:
            memory_store: Anomaly memory store
            min_similarity_threshold: Minimum similarity threshold
        """
        self._memory_store = memory_store
        self._min_similarity_threshold = min_similarity_threshold
    
    def retrieve(
        self,
        sensor_id: int,
        ml_features: Dict[str, Any],
        regime: str,
        top_k: int = 5,
        sensor_type: Optional[str] = None,
        time_window: Optional[tuple] = None,
    ) -> List[MemoryEvent]:
        """
        Retrieve similar historical events.
        
        Args:
            sensor_id: Sensor identifier
            ml_features: ML features dictionary
            regime: Current operational regime
            top_k: Number of results
            sensor_type: Filter by sensor type
            time_window: (start, end) timestamp window
        
        Returns:
            List of similar MemoryEvents
        """
        # 1. Generate query embedding
        query_text = self._build_query_text(sensor_id, ml_features, regime)
        query_embedding = self._memory_store._generate_embedding(query_text)
        
        # 2. Retrieve from memory store
        similar_events = self._memory_store.retrieve_similar(
            query_embedding=query_embedding,
            sensor_id=sensor_id,
            regime=regime,
            sensor_type=sensor_type,
            top_k=top_k,
            time_window=time_window,
        )
        
        # 3. Filter by operational similarity
        filtered = self._filter_operational_similarity(similar_events, ml_features)
        
        # 4. Top-k
        return filtered[:top_k]
    
    def _build_query_text(
        self,
        sensor_id: int,
        ml_features: Dict[str, Any],
        regime: str,
    ) -> str:
        """
        Build query text for embedding.
        
        Args:
            sensor_id: Sensor identifier
            ml_features: ML features dictionary
            regime: Current operational regime
        
        Returns:
            Query text
        """
        current_value = ml_features.get("current_value", 0.0)
        baseline = ml_features.get("baseline", 0.0)
        z_score = ml_features.get("z_score", 0.0)
        
        text = f"Sensor {sensor_id} en régimen {regime}. "
        text += f"Valor actual: {current_value:.2f}, Baseline: {baseline:.2f}, Z-score: {z_score:.2f}."
        
        # Add dynamic features if available
        dynamic_features = ml_features.get("dynamic_features", {})
        if dynamic_features:
            derivative = dynamic_features.get("derivative")
            if derivative is not None:
                text += f" Derivada: {derivative:.2f}."
            
            rolling_std = dynamic_features.get("rolling_std_1h")
            if rolling_std is not None:
                text += f" Volatilidad: {rolling_std:.2f}."
        
        return text
    
    def _filter_operational_similarity(
        self,
        events: List[MemoryEvent],
        current_features: Dict[str, Any],
    ) -> List[MemoryEvent]:
        """
        Filter events by operational similarity.
        
        Args:
            events: List of events to filter
            current_features: Current ML features
        
        Returns:
            Filtered list of events
        """
        filtered = []
        current_value = current_features.get("current_value", 0.0)
        
        for event in events:
            # Check value similarity (within 3σ)
            event_value = event.metadata.get("value", 0.0)
            value_diff = abs(event_value - current_value)
            
            # Use baseline as reference for σ
            baseline = current_features.get("baseline", 1.0)
            if baseline == 0:
                baseline = 1.0
            
            if value_diff > 3 * baseline:
                continue  # Too different
            
            filtered.append(event)
        
        return filtered
