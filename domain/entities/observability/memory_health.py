"""
MemoryHealth domain entity for memory health monitoring.

This is a domain entity for monitoring memory health and quality.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass(frozen=True)
class MemoryHealth:
    """
    Domain entity for memory health (DDD value object).
    
    This represents the domain concept of memory health and quality.
    """
    
    timestamp: float
    memory_quality_score: float
    retrieval_usefulness_score: float
    semantic_duplication_rate: float
    stale_memory_rate: float
    low_quality_memory_rate: float
    embedding_repetition_rate: float
    retrieval_degradation_score: float
    memory_explosion_risk: float
    cleanup_recommendations: List[str]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "memory_quality_score": self.memory_quality_score,
            "retrieval_usefulness_score": self.retrieval_usefulness_score,
            "semantic_duplication_rate": self.semantic_duplication_rate,
            "stale_memory_rate": self.stale_memory_rate,
            "low_quality_memory_rate": self.low_quality_memory_rate,
            "embedding_repetition_rate": self.embedding_repetition_rate,
            "retrieval_degradation_score": self.retrieval_degradation_score,
            "memory_explosion_risk": self.memory_explosion_risk,
            "cleanup_recommendations": self.cleanup_recommendations,
        }
