"""
MemoryHealthMonitor for monitoring memory health and quality.

Detects semantic duplication, stale memory, low-quality memory, embedding repetition, retrieval degradation, and memory explosion risk.
"""

from typing import List, Dict, Any
import time

from domain.entities.observability import MemoryHealth


class MemoryHealthMonitor:
    """Monitor for memory health and quality."""
    
    def __init__(self):
        """Initialize memory health monitor."""
        self._semantic_duplication_samples: List[float] = []
        self._stale_memory_samples: List[float] = []
        self._low_quality_samples: List[float] = []
        self._embedding_repetition_samples: List[float] = []
        self._retrieval_degradation_samples: List[float] = []
        self._memory_explosion_samples: List[float] = []
    
    def record_semantic_duplication(self, duplication_rate: float) -> None:
        """Record semantic duplication rate."""
        self._semantic_duplication_samples.append(duplication_rate)
    
    def record_stale_memory(self, stale_rate: float) -> None:
        """Record stale memory rate."""
        self._stale_memory_samples.append(stale_rate)
    
    def record_low_quality_memory(self, low_quality_rate: float) -> None:
        """Record low-quality memory rate."""
        self._low_quality_samples.append(low_quality_rate)
    
    def record_embedding_repetition(self, repetition_rate: float) -> None:
        """Record embedding repetition rate."""
        self._embedding_repetition_samples.append(repetition_rate)
    
    def record_retrieval_degradation(self, degradation_score: float) -> None:
        """Record retrieval degradation score."""
        self._retrieval_degradation_samples.append(degradation_score)
    
    def record_memory_explosion_risk(self, explosion_risk: float) -> None:
        """Record memory explosion risk."""
        self._memory_explosion_samples.append(explosion_risk)
    
    def assess_health(self) -> MemoryHealth:
        """Assess current memory health."""
        # Calculate individual rates
        semantic_duplication_rate = self._calculate_mean(self._semantic_duplication_samples)
        stale_memory_rate = self._calculate_mean(self._stale_memory_samples)
        low_quality_memory_rate = self._calculate_mean(self._low_quality_samples)
        embedding_repetition_rate = self._calculate_mean(self._embedding_repetition_samples)
        retrieval_degradation_score = self._calculate_mean(self._retrieval_degradation_samples)
        memory_explosion_risk = self._calculate_mean(self._memory_explosion_samples)
        
        # Calculate composite scores
        memory_quality_score = self._calculate_memory_quality_score(
            semantic_duplication_rate,
            stale_memory_rate,
            low_quality_memory_rate,
        )
        
        retrieval_usefulness_score = 1.0 - retrieval_degradation_score
        
        # Generate cleanup recommendations
        cleanup_recommendations = self._generate_cleanup_recommendations(
            semantic_duplication_rate,
            stale_memory_rate,
            low_quality_memory_rate,
            memory_explosion_risk,
        )
        
        return MemoryHealth(
            timestamp=time.time(),
            memory_quality_score=memory_quality_score,
            retrieval_usefulness_score=retrieval_usefulness_score,
            semantic_duplication_rate=semantic_duplication_rate,
            stale_memory_rate=stale_memory_rate,
            low_quality_memory_rate=low_quality_memory_rate,
            embedding_repetition_rate=embedding_repetition_rate,
            retrieval_degradation_score=retrieval_degradation_score,
            memory_explosion_risk=memory_explosion_risk,
            cleanup_recommendations=cleanup_recommendations,
        )
    
    def _calculate_mean(self, samples: List[float]) -> float:
        """Calculate mean of samples."""
        return sum(samples) / len(samples) if samples else 0.0
    
    def _calculate_memory_quality_score(
        self,
        semantic_duplication: float,
        stale_memory: float,
        low_quality: float,
    ) -> float:
        """Calculate memory quality score."""
        # Quality score decreases with duplication, stale memory, and low quality
        quality_score = 1.0 - (semantic_duplication + stale_memory + low_quality) / 3.0
        return max(0.0, min(1.0, quality_score))
    
    def _generate_cleanup_recommendations(
        self,
        semantic_duplication: float,
        stale_memory: float,
        low_quality: float,
        explosion_risk: float,
    ) -> List[str]:
        """Generate cleanup recommendations."""
        recommendations = []
        
        if semantic_duplication > 0.1:
            recommendations.append("Consider deduplicating semantically similar events")
        
        if stale_memory > 0.2:
            recommendations.append("Review and clean up stale memory events")
        
        if low_quality > 0.15:
            recommendations.append("Filter out low-quality memory events")
        
        if explosion_risk > 0.7:
            recommendations.append("URGENT: Memory explosion risk detected - review TTL policies")
        
        if not recommendations:
            recommendations.append("Memory health is healthy - no cleanup needed")
        
        return recommendations
    
    def reset(self) -> None:
        """Reset all samples."""
        self._semantic_duplication_samples = []
        self._stale_memory_samples = []
        self._low_quality_samples = []
        self._embedding_repetition_samples = []
        self._retrieval_degradation_samples = []
        self._memory_explosion_samples = []
