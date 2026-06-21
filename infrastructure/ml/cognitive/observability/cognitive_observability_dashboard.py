"""
CognitiveObservabilityDashboard for providing observability metrics.

Provides metrics for regime distribution, anomaly distribution, retrieval hit-rate, retrieval similarity distribution, explainability consistency, confidence distribution, memory growth rate, TTL cleanup rate, and contextual confidence calibration.
"""

from typing import Dict, Any
import time

from domain.entities.observability import CognitiveMetrics, MemoryHealth, DriftResult
from .cognitive_metrics_collector import CognitiveMetricsCollector
from .memory_health_monitor import MemoryHealthMonitor
from .drift_detection_engine import DriftDetectionEngine
from .explainability_validator import ExplainabilityValidator
from .feedback_loop_manager import FeedbackLoopManager


class CognitiveObservabilityDashboard:
    """Dashboard for cognitive observability metrics."""
    
    def __init__(
        self,
        metrics_collector: CognitiveMetricsCollector,
        memory_health_monitor: MemoryHealthMonitor,
        drift_detection_engine: DriftDetectionEngine,
        explainability_validator: ExplainabilityValidator,
        feedback_loop_manager: FeedbackLoopManager,
    ):
        """
        Initialize observability dashboard.
        
        Args:
            metrics_collector: Cognitive metrics collector
            memory_health_monitor: Memory health monitor
            drift_detection_engine: Drift detection engine
            explainability_validator: Explainability validator
            feedback_loop_manager: Feedback loop manager
        """
        self._metrics_collector = metrics_collector
        self._memory_health_monitor = memory_health_monitor
        self._drift_detection_engine = drift_detection_engine
        self._explainability_validator = explainability_validator
        self._feedback_loop_manager = feedback_loop_manager
    
    def get_observability_summary(self) -> Dict[str, Any]:
        """Get comprehensive observability summary."""
        # Collect current metrics
        cognitive_metrics = self._metrics_collector.collect_metrics()
        
        # Assess memory health
        memory_health = self._memory_health_monitor.assess_health()
        
        # Get feedback summary
        alert_feedback = self._feedback_loop_manager.get_feedback_summary("alert_usefulness")
        retrieval_feedback = self._feedback_loop_manager.get_feedback_summary("retrieval_usefulness")
        explainability_feedback = self._feedback_loop_manager.get_feedback_summary("explainability_usefulness")
        
        return {
            "timestamp": time.time(),
            "cognitive_metrics": cognitive_metrics.to_dict(),
            "memory_health": memory_health.to_dict(),
            "feedback_summary": {
                "alert_usefulness": alert_feedback,
                "retrieval_usefulness": retrieval_feedback,
                "explainability_usefulness": explainability_feedback,
            },
            "health_status": self._calculate_health_status(cognitive_metrics, memory_health),
        }
    
    def get_regime_distribution(self) -> Dict[str, int]:
        """Get current regime distribution."""
        metrics = self._metrics_collector.collect_metrics()
        return metrics.regime_distribution
    
    def get_anomaly_distribution(self) -> Dict[str, int]:
        """Get current anomaly distribution."""
        metrics = self._metrics_collector.collect_metrics()
        return metrics.anomaly_distribution
    
    def get_retrieval_metrics(self) -> Dict[str, float]:
        """Get retrieval metrics."""
        metrics = self._metrics_collector.collect_metrics()
        return {
            "hit_rate": metrics.retrieval_hit_rate,
            "similarity_mean": metrics.retrieval_similarity_mean,
        }
    
    def get_explainability_metrics(self) -> Dict[str, float]:
        """Get explainability metrics."""
        metrics = self._metrics_collector.collect_metrics()
        return {
            "consistency": metrics.explainability_consistency,
        }
    
    def get_confidence_distribution(self) -> Dict[str, float]:
        """Get confidence distribution."""
        metrics = self._metrics_collector.collect_metrics()
        return metrics.confidence_distribution
    
    def get_memory_metrics(self) -> Dict[str, float]:
        """Get memory metrics."""
        metrics = self._metrics_collector.collect_metrics()
        return {
            "growth_rate": metrics.memory_growth_rate,
            "ttl_cleanup_rate": metrics.ttl_cleanup_rate,
        }
    
    def get_contextual_confidence_calibration(self) -> float:
        """Get contextual confidence calibration."""
        metrics = self._metrics_collector.collect_metrics()
        return metrics.contextual_confidence_calibration
    
    def _calculate_health_status(
        self,
        cognitive_metrics: CognitiveMetrics,
        memory_health: MemoryHealth,
    ) -> str:
        """Calculate overall health status."""
        # Check memory health
        if memory_health.memory_explosion_risk > 0.7:
            return "CRITICAL"
        
        if memory_health.memory_quality_score < 0.5:
            return "WARNING"
        
        # Check retrieval health
        if cognitive_metrics.retrieval_hit_rate < 0.5:
            return "WARNING"
        
        # Check explainability health
        if cognitive_metrics.explainability_consistency < 0.7:
            return "WARNING"
        
        return "HEALTHY"
