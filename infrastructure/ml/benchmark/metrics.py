"""Benchmark metrics — compute performance metrics for anomaly detection.

Single responsibility: calculate precision, recall, F1, FPR, and detection delay
from predictions and ground truth labels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MetricsResult:
    """Benchmark metrics result.
    
    Attributes:
        precision: TP / (TP + FP), range [0, 1].
        recall: TP / (TP + FN), range [0, 1].
        f1_score: Harmonic mean of precision and recall.
        false_positive_rate: FP / (FP + TN), range [0, 1].
        avg_detection_delay: Average delay in detecting anomalies (samples).
        true_positives: Count of true positives.
        false_positives: Count of false positives.
        true_negatives: Count of true negatives.
        false_negatives: Count of false negatives.
    """
    precision: float
    recall: float
    f1_score: float
    false_positive_rate: float
    avg_detection_delay: Optional[float]
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "false_positive_rate": round(self.false_positive_rate, 4),
            "avg_detection_delay": round(self.avg_detection_delay, 2) if self.avg_detection_delay else None,
            "confusion_matrix": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "true_negatives": self.true_negatives,
                "false_negatives": self.false_negatives,
            },
        }


class BenchmarkMetrics:
    """Computes benchmark metrics for anomaly detection.
    
    Metrics:
    - Precision: TP / (TP + FP)
    - Recall: TP / (TP + FN)
    - F1-score: 2 * (precision * recall) / (precision + recall)
    - False Positive Rate: FP / (FP + TN)
    - Detection Delay: Average time to detect anomaly after it starts
    """
    
    @staticmethod
    def compute(
        predictions: List[bool],
        ground_truth: List[bool],
        compute_delay: bool = True,
    ) -> MetricsResult:
        """Compute benchmark metrics.
        
        Args:
            predictions: List of predicted labels (True = anomaly).
            ground_truth: List of ground truth labels (True = anomaly).
            compute_delay: Whether to compute detection delay.
        
        Returns:
            MetricsResult with all computed metrics.
        
        Raises:
            ValueError: If predictions and ground_truth have different lengths.
        """
        if len(predictions) != len(ground_truth):
            raise ValueError(
                f"Predictions ({len(predictions)}) and ground truth ({len(ground_truth)}) "
                f"must have same length"
            )
        
        # Compute confusion matrix
        tp = sum(1 for p, g in zip(predictions, ground_truth) if p and g)
        fp = sum(1 for p, g in zip(predictions, ground_truth) if p and not g)
        tn = sum(1 for p, g in zip(predictions, ground_truth) if not p and not g)
        fn = sum(1 for p, g in zip(predictions, ground_truth) if not p and g)
        
        # Compute precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Compute recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # Compute F1-score
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        
        # Compute False Positive Rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # Compute detection delay
        avg_delay = None
        if compute_delay:
            avg_delay = BenchmarkMetrics._compute_detection_delay(
                predictions,
                ground_truth,
            )
        
        logger.info(
            "benchmark_metrics_computed",
            extra={
                "event": "METRICS_COMPUTED",
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4),
                "fpr": round(fpr, 4),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
            },
        )
        
        return MetricsResult(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            false_positive_rate=fpr,
            avg_detection_delay=avg_delay,
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
        )
    
    @staticmethod
    def _compute_detection_delay(
        predictions: List[bool],
        ground_truth: List[bool],
    ) -> Optional[float]:
        """Compute average detection delay.
        
        Detection delay = number of samples between anomaly start and
        first detection.
        
        Args:
            predictions: List of predicted labels.
            ground_truth: List of ground truth labels.
        
        Returns:
            Average detection delay in samples, or None if no anomalies detected.
        """
        delays: List[int] = []
        
        # Find anomaly segments in ground truth
        in_anomaly = False
        anomaly_start = 0
        
        for i, (pred, truth) in enumerate(zip(predictions, ground_truth)):
            # Anomaly starts
            if truth and not in_anomaly:
                in_anomaly = True
                anomaly_start = i
            
            # Anomaly detected
            if in_anomaly and pred:
                delay = i - anomaly_start
                delays.append(delay)
                in_anomaly = False  # Reset for next anomaly
            
            # Anomaly ends without detection
            if not truth and in_anomaly:
                in_anomaly = False
        
        if not delays:
            return None
        
        return sum(delays) / len(delays)
