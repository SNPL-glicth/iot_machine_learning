"""Benchmark subsystem — performance evaluation for anomaly detection.

Provides dataset loading, metrics computation, and benchmark execution
for evaluating the ZENIN pipeline against labeled datasets.
"""

from .dataset_loader import DatasetLoader, DatasetSample
from .metrics import BenchmarkMetrics, MetricsResult
from .benchmark_runner import BenchmarkRunner, BenchmarkReport
from .empirical_validation_harness import EmpiricalBenchmarkReport, EmpiricalValidationHarness

__all__ = [
    "DatasetLoader",
    "DatasetSample",
    "BenchmarkMetrics",
    "MetricsResult",
    "BenchmarkRunner",
    "BenchmarkReport",
    "EmpiricalBenchmarkReport",
    "EmpiricalValidationHarness",
]
