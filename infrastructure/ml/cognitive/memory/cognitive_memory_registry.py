"""
CognitiveMemoryRegistry for managing memory configuration.

Provides configuration for TTL, quality thresholds, and feature flags.
"""

from threading import RLock
from typing import Dict


class CognitiveMemoryRegistry:
    """Registry for cognitive memory configuration."""
    
    def __init__(self):
        """Initialize registry with default configuration."""
        self._lock = RLock()
        
        # TTL configuration (seconds)
        self._ttl_config = {
            "ANOMALY_CONFIRMED": 365 * 24 * 3600,  # 1 año
            "ANOMALY_SUSPECTED": 7 * 24 * 3600,  # 1 semana
            "REGIME_TRANSITION": 180 * 24 * 3600,  # 6 meses
            "OPERATIONAL_TRANSITION": 90 * 24 * 3600,  # 3 meses
            "OPERATIONAL_STATE": 30 * 24 * 3600,  # 1 mes
        }
        
        # Quality thresholds
        self._min_anomaly_score = 0.6
        self._min_feature_variability = 0.1
        
        # Feature flags
        self._enable_memory = True
        self._enable_retrieval = True
    
    def get_ttl(self, event_type: str) -> int:
        """
        Get TTL for event type.
        
        Args:
            event_type: Event type
        
        Returns:
            TTL in seconds
        """
        with self._lock:
            return self._ttl_config.get(event_type, 7 * 24 * 3600)  # Default: 1 semana
    
    def set_ttl(self, event_type: str, ttl: int) -> None:
        """
        Set TTL for event type.
        
        Args:
            event_type: Event type
            ttl: TTL in seconds
        """
        with self._lock:
            self._ttl_config[event_type] = ttl
    
    @property
    def min_anomaly_score(self) -> float:
        """Minimum anomaly score for persistence."""
        with self._lock:
            return self._min_anomaly_score
    
    @property
    def min_feature_variability(self) -> float:
        """Minimum feature variability (CV) for persistence."""
        with self._lock:
            return self._min_feature_variability
    
    @property
    def enable_memory(self) -> bool:
        """Whether memory storage is enabled."""
        with self._lock:
            return self._enable_memory
    
    @property
    def enable_retrieval(self) -> bool:
        """Whether retrieval is enabled."""
        with self._lock:
            return self._enable_retrieval
    
    def enable_memory_storage(self, enabled: bool) -> None:
        """
        Enable or disable memory storage.
        
        Args:
            enabled: Whether to enable storage
        """
        with self._lock:
            self._enable_memory = enabled
    
    def enable_retrieval_feature(self, enabled: bool) -> None:
        """
        Enable or disable retrieval.
        
        Args:
            enabled: Whether to enable retrieval
        """
        with self._lock:
            self._enable_retrieval = enabled
    
    def get_ttl_config(self) -> Dict[str, int]:
        """Get all TTL configuration."""
        with self._lock:
            return self._ttl_config.copy()
