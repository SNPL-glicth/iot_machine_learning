"""Benchmark subsystem — performance evaluation for anomaly detection.

Provides dataset loading, metrics computation, and benchmark execution
for evaluating the ZENIN pipeline against labeled datasets.
"""

from .dataset_loader import DatasetLoader, DatasetSample
from .metrics import BenchmarkMetrics, MetricsResult
from .benchmark_runner import BenchmarkRunner, BenchmarkReport

__all__ = [
    "DatasetLoader",
    "DatasetSample",
    "BenchmarkMetrics",
    "MetricsResult",
    "BenchmarkRunner",
    "BenchmarkReport",
]
