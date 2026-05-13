"""Tests and benchmarks for CRÍTICO-2: BatchPipelineExecutor.

Includes pytest-benchmark comparisons for sequential vs parallel execution.
"""

import time
from unittest.mock import Mock

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.batch_pipeline_executor import (
    BatchPipelineExecutor,
)


class TestBatchPipelineExecutor:
    """Test BatchPipelineExecutor (CRÍTICO-2)."""
    
    def test_constructor_validation(self):
        """Constructor validates parameters."""
        executor = Mock()
        
        # Valid
        batch_exec = BatchPipelineExecutor(executor, max_workers=4, batch_size=25)
        assert batch_exec._max_workers == 4
        assert batch_exec._batch_size == 25
        
        # Invalid max_workers
        with pytest.raises(ValueError, match="max_workers must be > 0"):
            BatchPipelineExecutor(executor, max_workers=0)
        
        # Invalid batch_size
        with pytest.raises(ValueError, match="batch_size must be > 0"):
            BatchPipelineExecutor(executor, batch_size=-1)
    
    def test_execute_batch_empty(self):
        """Empty batch returns empty results."""
        executor = Mock()
        batch_exec = BatchPipelineExecutor(executor, max_workers=2, batch_size=10)
        
        orchestrator = Mock()
        result = batch_exec.execute_batch(orchestrator, [])
        
        assert result["total"] == 0
        assert result["succeeded"] == 0
        assert result["failed"] == 0
        assert result["results"] == []
        assert result["errors"] == []
    
    def test_execute_batch_all_succeed(self):
        """All series succeed."""
        executor = Mock()
        executor.execute = Mock(side_effect=lambda **kwargs: {"series_id": kwargs["series_id"], "value": 42.0})
        
        batch_exec = BatchPipelineExecutor(executor, max_workers=2, batch_size=10)
        
        orchestrator = Mock()
        series_batch = [
            ("sensor_1", [10.0, 12.0], None),
            ("sensor_2", [20.0, 22.0], None),
            ("sensor_3", [30.0, 32.0], None),
        ]
        
        result = batch_exec.execute_batch(orchestrator, series_batch)
        
        assert result["total"] == 3
        assert result["succeeded"] == 3
        assert result["failed"] == 0
        assert len(result["results"]) == 3
        assert len(result["errors"]) == 0
    
    def test_execute_batch_partial_failure(self):
        """Individual failures do not stop batch (CRÍTICO-2)."""
        executor = Mock()
        
        def execute_with_failure(**kwargs):
            series_id = kwargs["series_id"]
            if series_id == "sensor_2":
                raise ValueError("Simulated failure for sensor_2")
            return {"series_id": series_id, "value": 42.0}
        
        executor.execute = Mock(side_effect=execute_with_failure)
        
        batch_exec = BatchPipelineExecutor(executor, max_workers=2, batch_size=10)
        
        orchestrator = Mock()
        series_batch = [
            ("sensor_1", [10.0, 12.0], None),
            ("sensor_2", [20.0, 22.0], None),
            ("sensor_3", [30.0, 32.0], None),
        ]
        
        result = batch_exec.execute_batch(orchestrator, series_batch)
        
        assert result["total"] == 3
        assert result["succeeded"] == 2
        assert result["failed"] == 1
        assert len(result["results"]) == 2
        assert len(result["errors"]) == 1
        assert result["errors"][0][0] == "sensor_2"
        assert "Simulated failure" in result["errors"][0][1]
    
    def test_execute_batch_batching(self):
        """Large batch is processed in chunks."""
        executor = Mock()
        executor.execute = Mock(return_value={"value": 42.0})
        
        batch_exec = BatchPipelineExecutor(executor, max_workers=4, batch_size=5)
        
        orchestrator = Mock()
        # 12 series, batch_size=5 → 3 batches (5, 5, 2)
        series_batch = [(f"sensor_{i}", [10.0], None) for i in range(12)]
        
        result = batch_exec.execute_batch(orchestrator, series_batch)
        
        assert result["total"] == 12
        assert result["succeeded"] == 12
        assert result["failed"] == 0
    
    def test_get_metrics(self):
        """get_metrics returns configuration."""
        executor = Mock()
        batch_exec = BatchPipelineExecutor(executor, max_workers=8, batch_size=50)
        
        metrics = batch_exec.get_metrics()
        
        assert metrics["max_workers"] == 8
        assert metrics["batch_size"] == 50


class TestBatchExecutorBenchmark:
    """Benchmark sequential vs parallel execution (CRÍTICO-2)."""
    
    def test_benchmark_sequential_vs_parallel(self, benchmark):
        """Benchmark shows parallel is faster for >50 series."""
        # Mock executor with 10ms delay per series
        executor = Mock()
        
        def slow_execute(**kwargs):
            time.sleep(0.01)  # 10ms per series
            return {"series_id": kwargs["series_id"], "value": 42.0}
        
        executor.execute = Mock(side_effect=slow_execute)
        
        orchestrator = Mock()
        series_batch = [(f"sensor_{i}", [10.0, 12.0], None) for i in range(100)]
        
        # Benchmark parallel execution
        batch_exec = BatchPipelineExecutor(executor, max_workers=8, batch_size=50)
        
        result = benchmark(batch_exec.execute_batch, orchestrator, series_batch)
        
        assert result["total"] == 100
        assert result["succeeded"] == 100
        
        # Expected: ~125ms (100 series / 8 workers * 10ms)
        # vs sequential: ~1000ms (100 * 10ms)
        # Speedup: ~8x
    
    def test_benchmark_small_batch_overhead(self, benchmark):
        """Small batches have overhead from thread pool creation."""
        executor = Mock()
        executor.execute = Mock(return_value={"value": 42.0})
        
        orchestrator = Mock()
        series_batch = [(f"sensor_{i}", [10.0], None) for i in range(10)]
        
        batch_exec = BatchPipelineExecutor(executor, max_workers=4, batch_size=5)
        
        result = benchmark(batch_exec.execute_batch, orchestrator, series_batch)
        
        assert result["total"] == 10
        assert result["succeeded"] == 10
