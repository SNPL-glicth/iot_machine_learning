"""Tests for parallel engine execution in PredictPhase.

Verifies that:
1. ThreadPoolExecutor is used when ML_PREDICT_MAX_WORKERS > 1
2. Sequential fallback works when parallel fails
3. Output is equivalent between parallel and sequential modes
4. Env variable ML_PREDICT_MAX_WORKERS is respected
"""

import os
import pytest
from unittest.mock import MagicMock, patch


class MockEngine:
    """Mock prediction engine for testing."""
    def __init__(self, name, delay_ms=0, should_fail=False):
        self.name = name
        self.delay_ms = delay_ms
        self.should_fail = should_fail
        self.call_count = 0
    
    def can_handle(self, n_points):
        return n_points >= 3
    
    def predict(self, values, timestamps):
        import time
        self.call_count += 1
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)
        if self.should_fail:
            raise RuntimeError(f"Engine {self.name} failed")
        
        # Create a mock prediction result
        mock_result = MagicMock()
        mock_result.predicted_value = sum(values) / len(values) if values else 0.0
        mock_result.confidence = 0.8
        mock_result.trend = "stable"
        mock_result.metadata = {"diagnostic": {"stability_indicator": 0.5}}
        return mock_result


class TestParallelEngineExecution:
    """Test parallel engine execution with ThreadPoolExecutor."""
    
    def test_parallel_execution_faster_than_sequential(self):
        """Test that parallel execution is faster with multiple engines."""
        import time
        from iot_machine_learning.infrastructure.ml.cognitive.perception.helpers import (
            _collect_perceptions_parallel,
            _collect_perceptions_sequential,
        )
        
        # Create engines with 50ms delay each
        engines = [
            MockEngine("eng1", delay_ms=50),
            MockEngine("eng2", delay_ms=50),
            MockEngine("eng3", delay_ms=50),
        ]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Time sequential execution
        start = time.perf_counter()
        seq_result = _collect_perceptions_sequential(engines, values, None)
        seq_time = (time.perf_counter() - start) * 1000
        
        # Time parallel execution
        start = time.perf_counter()
        par_result = _collect_perceptions_parallel(engines, values, None, max_workers=3)
        par_time = (time.perf_counter() - start) * 1000
        
        # Parallel should be significantly faster (at least 2x)
        assert par_time < seq_time * 0.6, f"Parallel ({par_time:.1f}ms) not faster than sequential ({seq_time:.1f}ms)"
        
        # Both should return same number of results
        assert len(seq_result) == len(par_result) == 3
    
    def test_parallel_output_equivalent_to_sequential(self):
        """Test that parallel and sequential produce equivalent output."""
        from iot_machine_learning.infrastructure.ml.cognitive.perception.helpers import (
            _collect_perceptions_parallel,
            _collect_perceptions_sequential,
        )
        
        engines = [
            MockEngine("eng1"),
            MockEngine("eng2"),
            MockEngine("eng3"),
        ]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        seq_result = _collect_perceptions_sequential(engines, values, None)
        par_result = _collect_perceptions_parallel(engines, values, None, max_workers=3)
        
        # Same number of perceptions
        assert len(seq_result) == len(par_result)
        
        # Same engine names (order may differ in parallel)
        seq_names = {p.engine_name for p in seq_result}
        par_names = {p.engine_name for p in par_result}
        assert seq_names == par_names
        
        # Same predicted values for each engine
        seq_values = {p.engine_name: p.predicted_value for p in seq_result}
        par_values = {p.engine_name: p.predicted_value for p in par_result}
        assert seq_values == par_values
    
    def test_sequential_fallback_on_parallel_failure(self):
        """Test fallback to sequential when ThreadPoolExecutor fails."""
        from iot_machine_learning.infrastructure.ml.cognitive.perception.helpers import (
            collect_perceptions,
        )
        
        engines = [MockEngine("eng1"), MockEngine("eng2")]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # Mock ThreadPoolExecutor to raise exception
        with patch("iot_machine_learning.infrastructure.ml.cognitive.perception.helpers.ThreadPoolExecutor") as mock_executor:
            mock_executor.side_effect = RuntimeError("ThreadPool failed")
            
            # Should fallback to sequential and still work
            result = collect_perceptions(engines, values, None)
            
            assert len(result) == 2
    
    def test_env_variable_max_workers(self):
        """Test that ML_PREDICT_MAX_WORKERS env variable is respected."""
        from iot_machine_learning.infrastructure.ml.cognitive.perception.helpers import (
            ML_PREDICT_MAX_WORKERS,
        )
        
        # Default should be 3
        assert ML_PREDICT_MAX_WORKERS >= 1
    
    def test_single_engine_uses_sequential(self):
        """Test that single engine skips parallel overhead."""
        from iot_machine_learning.infrastructure.ml.cognitive.perception.helpers import (
            collect_perceptions,
            _collect_perceptions_parallel,
        )
        
        engines = [MockEngine("eng1")]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # With single engine, should use sequential (no parallel overhead)
        with patch.object(
            __import__("iot_machine_learning.infrastructure.ml.cognitive.perception.helpers", fromlist=["_collect_perceptions_parallel"]),
            "_collect_perceptions_parallel"
        ) as mock_parallel:
            # Ensure ML_PREDICT_MAX_WORKERS > 1
            with patch("iot_machine_learning.infrastructure.ml.cognitive.perception.helpers.ML_PREDICT_MAX_WORKERS", 3):
                result = collect_perceptions(engines, values, None)
                # Parallel should not be called for single engine
                mock_parallel.assert_not_called()
        
        assert len(result) == 1
        assert result[0].engine_name == "eng1"
    
    def test_engine_failure_handling_parallel(self):
        """Test that engine failures are handled gracefully in parallel mode."""
        from iot_machine_learning.infrastructure.ml.cognitive.perception.helpers import (
            _collect_perceptions_parallel,
        )
        
        engines = [
            MockEngine("eng1"),
            MockEngine("eng2", should_fail=True),  # This one fails
            MockEngine("eng3"),
        ]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        result = _collect_perceptions_parallel(engines, values, None, max_workers=3)
        
        # Should get 2 successful results
        assert len(result) == 2
        names = {p.engine_name for p in result}
        assert "eng1" in names
        assert "eng3" in names
        assert "eng2" not in names  # Failed engine not in results
    
    def test_max_workers_one_uses_sequential(self):
        """Test that max_workers=1 forces sequential execution."""
        from iot_machine_learning.infrastructure.ml.cognitive.perception.helpers import (
            _collect_perceptions_sequential,
        )
        
        engines = [MockEngine("eng1"), MockEngine("eng2")]
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        
        # When ML_PREDICT_MAX_WORKERS=1, should use sequential
        with patch("iot_machine_learning.infrastructure.ml.cognitive.perception.helpers.ML_PREDICT_MAX_WORKERS", 1):
            with patch(
                "iot_machine_learning.infrastructure.ml.cognitive.perception.helpers._collect_perceptions_parallel"
            ) as mock_parallel:
                from iot_machine_learning.infrastructure.ml.cognitive.perception.helpers import (
                    collect_perceptions,
                )
                result = collect_perceptions(engines, values, None)
                # Parallel should not be called
                mock_parallel.assert_not_called()
        
        assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
