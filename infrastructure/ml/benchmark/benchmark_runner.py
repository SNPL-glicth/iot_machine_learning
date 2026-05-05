"""Benchmark runner — execute anomaly detection pipeline on labeled datasets.

Single responsibility: run pipeline on benchmark datasets, collect predictions,
and compute performance metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Callable, Any

from .dataset_loader import DatasetLoader, DatasetSample
from .metrics import BenchmarkMetrics, MetricsResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BenchmarkReport:
    """Benchmark execution report.
    
    Attributes:
        dataset_name: Name of the dataset.
        n_samples: Total number of samples.
        n_anomalies: Number of anomalies in ground truth.
        metrics: Computed metrics.
        decision_threshold: Threshold used for anomaly decision.
    """
    dataset_name: str
    n_samples: int
    n_anomalies: int
    metrics: MetricsResult
    decision_threshold: float
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset": self.dataset_name,
            "n_samples": self.n_samples,
            "n_anomalies": self.n_anomalies,
            "decision_threshold": self.decision_threshold,
            **self.metrics.to_dict(),
        }


class BenchmarkRunner:
    """Runs anomaly detection pipeline on benchmark datasets.
    
    Executes pipeline sequentially on each sample, collects predictions,
    and computes performance metrics against ground truth.
    
    Attributes:
        _decision_threshold: Confidence threshold for anomaly decision.
        _window_size: Number of historical samples to maintain.
    """
    
    def __init__(
        self,
        decision_threshold: float = 0.7,
        window_size: int = 50,
    ) -> None:
        """Initialize benchmark runner.
        
        Args:
            decision_threshold: Confidence threshold for anomaly (>= threshold).
            window_size: Number of historical samples for pipeline.
        
        Raises:
            ValueError: If decision_threshold not in [0, 1].
        """
        if not 0.0 <= decision_threshold <= 1.0:
            raise ValueError(
                f"decision_threshold must be in [0, 1], got {decision_threshold}"
            )
        
        self._decision_threshold = decision_threshold
        self._window_size = window_size
    
    def run(
        self,
        dataset_path: str,
        predict_fn: Callable[[List[float], List[float]], dict],
        dataset_format: str = "nab",
    ) -> BenchmarkReport:
        """Run benchmark on a dataset.
        
        Args:
            dataset_path: Path to dataset CSV file.
            predict_fn: Function that takes (values, timestamps) and returns
                prediction dict with 'fused_confidence' key.
            dataset_format: Dataset format ("nab" or "yahoo").
        
        Returns:
            BenchmarkReport with metrics.
        
        Raises:
            ValueError: If dataset_format is invalid.
        """
        # Load dataset
        if dataset_format == "nab":
            samples = DatasetLoader.load_nab(dataset_path)
        elif dataset_format == "yahoo":
            samples = DatasetLoader.load_yahoo(dataset_path)
        else:
            raise ValueError(f"Invalid dataset_format: {dataset_format}")
        
        dataset_name = Path(dataset_path).stem
        
        logger.info(
            "benchmark_started",
            extra={
                "event": "BENCHMARK_START",
                "dataset": dataset_name,
                "n_samples": len(samples),
                "decision_threshold": self._decision_threshold,
            },
        )
        
        # Run pipeline and collect predictions
        predictions: List[bool] = []
        ground_truth: List[bool] = []
        
        values_buffer: List[float] = []
        timestamps_buffer: List[float] = []
        
        for i, sample in enumerate(samples):
            # Add to buffer
            values_buffer.append(sample.value)
            timestamps_buffer.append(sample.timestamp)
            
            # Maintain window size
            if len(values_buffer) > self._window_size:
                values_buffer.pop(0)
                timestamps_buffer.pop(0)
            
            # Skip until we have enough samples
            if len(values_buffer) < 10:
                continue
            
            try:
                # Run prediction
                result = predict_fn(values_buffer.copy(), timestamps_buffer.copy())
                
                # Extract confidence
                confidence = result.get('fused_confidence', 0.0)
                
                # Make decision
                is_anomaly_pred = confidence >= self._decision_threshold
                
                predictions.append(is_anomaly_pred)
                ground_truth.append(sample.is_anomaly)
            
            except Exception as e:
                logger.error(
                    "benchmark_prediction_failed",
                    extra={
                        "event": "PREDICTION_ERROR",
                        "sample_idx": i,
                        "error": str(e),
                        "action_taken": "skip_sample",
                    },
                )
                continue
        
        # Compute metrics
        metrics = BenchmarkMetrics.compute(
            predictions=predictions,
            ground_truth=ground_truth,
            compute_delay=True,
        )
        
        report = BenchmarkReport(
            dataset_name=dataset_name,
            n_samples=len(samples),
            n_anomalies=sum(1 for s in samples if s.is_anomaly),
            metrics=metrics,
            decision_threshold=self._decision_threshold,
        )
        
        logger.info(
            "benchmark_completed",
            extra={
                "event": "BENCHMARK_COMPLETE",
                "dataset": dataset_name,
                "precision": round(metrics.precision, 4),
                "recall": round(metrics.recall, 4),
                "f1_score": round(metrics.f1_score, 4),
            },
        )
        
        return report
    
    def run_multiple(
        self,
        dataset_paths: List[str],
        predict_fn: Callable[[List[float], List[float]], dict],
        dataset_format: str = "nab",
    ) -> List[BenchmarkReport]:
        """Run benchmark on multiple datasets.
        
        Args:
            dataset_paths: List of paths to dataset CSV files.
            predict_fn: Prediction function.
            dataset_format: Dataset format ("nab" or "yahoo").
        
        Returns:
            List of BenchmarkReport instances.
        """
        reports: List[BenchmarkReport] = []
        
        for path in dataset_paths:
            try:
                report = self.run(
                    dataset_path=path,
                    predict_fn=predict_fn,
                    dataset_format=dataset_format,
                )
                reports.append(report)
            
            except Exception as e:
                logger.error(
                    "benchmark_dataset_failed",
                    extra={
                        "event": "DATASET_ERROR",
                        "dataset": path,
                        "error": str(e),
                        "action_taken": "skip_dataset",
                    },
                )
                continue
        
        return reports
    
    @staticmethod
    def aggregate_reports(reports: List[BenchmarkReport]) -> dict:
        """Aggregate metrics from multiple reports.
        
        Args:
            reports: List of benchmark reports.
        
        Returns:
            Dictionary with aggregated metrics.
        """
        if not reports:
            return {}
        
        # Compute averages
        avg_precision = sum(r.metrics.precision for r in reports) / len(reports)
        avg_recall = sum(r.metrics.recall for r in reports) / len(reports)
        avg_f1 = sum(r.metrics.f1_score for r in reports) / len(reports)
        avg_fpr = sum(r.metrics.false_positive_rate for r in reports) / len(reports)
        
        delays = [
            r.metrics.avg_detection_delay
            for r in reports
            if r.metrics.avg_detection_delay is not None
        ]
        avg_delay = sum(delays) / len(delays) if delays else None
        
        return {
            "n_datasets": len(reports),
            "avg_precision": round(avg_precision, 4),
            "avg_recall": round(avg_recall, 4),
            "avg_f1_score": round(avg_f1, 4),
            "avg_false_positive_rate": round(avg_fpr, 4),
            "avg_detection_delay": round(avg_delay, 2) if avg_delay else None,
            "datasets": [r.dataset_name for r in reports],
        }
