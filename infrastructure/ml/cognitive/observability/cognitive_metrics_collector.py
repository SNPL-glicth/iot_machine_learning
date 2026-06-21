"""
CognitiveMetricsCollector for centralizing cognitive metrics.

Collects metrics from retrieval, explainability, regimes, drift, anomaly routing, and confidence calibration.
"""

from typing import Dict, Any, List
from collections import Counter
import time

from domain.entities.observability import CognitiveMetrics


class CognitiveMetricsCollector:
    """Collector for cognitive metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self._regime_counts: Counter = Counter()
        self._anomaly_counts: Counter = Counter()
        self._retrieval_hits: int = 0
        self._retrieval_misses: int = 0
        self._retrieval_similarities: List[float] = []
        self._explainability_consistencies: List[float] = []
        self._confidence_values: List[float] = []
        self._memory_growth_samples: List[float] = []
        self._ttl_cleanup_samples: List[float] = []
        self._contextual_confidence_samples: List[float] = []
    
    def record_regime(self, regime: str) -> None:
        """Record regime occurrence."""
        self._regime_counts[regime] += 1
    
    def record_anomaly(self, anomaly_type: str) -> None:
        """Record anomaly occurrence."""
        self._anomaly_counts[anomaly_type] += 1
    
    def record_retrieval(self, hit: bool, similarity: float = 0.0) -> None:
        """Record retrieval result."""
        if hit:
            self._retrieval_hits += 1
            self._retrieval_similarities.append(similarity)
        else:
            self._retrieval_misses += 1
    
    def record_explainability_consistency(self, consistency: float) -> None:
        """Record explainability consistency."""
        self._explainability_consistencies.append(consistency)
    
    def record_confidence(self, confidence: float) -> None:
        """Record confidence value."""
        self._confidence_values.append(confidence)
    
    def record_memory_growth(self, growth_rate: float) -> None:
        """Record memory growth rate."""
        self._memory_growth_samples.append(growth_rate)
    
    def record_ttl_cleanup(self, cleanup_rate: float) -> None:
        """Record TTL cleanup rate."""
        self._ttl_cleanup_samples.append(cleanup_rate)
    
    def record_contextual_confidence(self, confidence: float) -> None:
        """Record contextual confidence."""
        self._contextual_confidence_samples.append(confidence)
    
    def collect_metrics(self) -> CognitiveMetrics:
        """Collect current metrics."""
        # Calculate retrieval hit rate
        total_retrievals = self._retrieval_hits + self._retrieval_misses
        retrieval_hit_rate = self._retrieval_hits / total_retrievals if total_retrievals > 0 else 0.0
        
        # Calculate retrieval similarity mean
        retrieval_similarity_mean = (
            sum(self._retrieval_similarities) / len(self._retrieval_similarities)
            if self._retrieval_similarities else 0.0
        )
        
        # Calculate explainability consistency
        explainability_consistency = (
            sum(self._explainability_consistencies) / len(self._explainability_consistencies)
            if self._explainability_consistencies else 0.0
        )
        
        # Calculate confidence distribution
        confidence_distribution = self._calculate_confidence_distribution()
        
        # Calculate memory growth rate
        memory_growth_rate = (
            sum(self._memory_growth_samples) / len(self._memory_growth_samples)
            if self._memory_growth_samples else 0.0
        )
        
        # Calculate TTL cleanup rate
        ttl_cleanup_rate = (
            sum(self._ttl_cleanup_samples) / len(self._ttl_cleanup_samples)
            if self._ttl_cleanup_samples else 0.0
        )
        
        # Calculate contextual confidence calibration
        contextual_confidence_calibration = (
            sum(self._contextual_confidence_samples) / len(self._contextual_confidence_samples)
            if self._contextual_confidence_samples else 0.0
        )
        
        return CognitiveMetrics(
            timestamp=time.time(),
            regime_distribution=dict(self._regime_counts),
            anomaly_distribution=dict(self._anomaly_counts),
            retrieval_hit_rate=retrieval_hit_rate,
            retrieval_similarity_mean=retrieval_similarity_mean,
            explainability_consistency=explainability_consistency,
            confidence_distribution=confidence_distribution,
            memory_growth_rate=memory_growth_rate,
            ttl_cleanup_rate=ttl_cleanup_rate,
            contextual_confidence_calibration=contextual_confidence_calibration,
        )
    
    def _calculate_confidence_distribution(self) -> Dict[str, float]:
        """Calculate confidence distribution buckets."""
        if not self._confidence_values:
            return {"low": 0.0, "medium": 0.0, "high": 0.0}
        
        low = sum(1 for c in self._confidence_values if c < 0.33)
        medium = sum(1 for c in self._confidence_values if 0.33 <= c < 0.66)
        high = sum(1 for c in self._confidence_values if c >= 0.66)
        
        total = len(self._confidence_values)
        return {
            "low": low / total,
            "medium": medium / total,
            "high": high / total,
        }
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._regime_counts = Counter()
        self._anomaly_counts = Counter()
        self._retrieval_hits = 0
        self._retrieval_misses = 0
        self._retrieval_similarities = []
        self._explainability_consistencies = []
        self._confidence_values = []
        self._memory_growth_samples = []
        self._ttl_cleanup_samples = []
        self._contextual_confidence_samples = []
