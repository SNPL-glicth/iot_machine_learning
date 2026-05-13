"""Batch Pipeline Executor for high-frequency sensor processing (CRÍTICO-2).

Processes multiple series in parallel using ThreadPoolExecutor for >500 sensors.

Applies SRP: BatchPipelineExecutor is separate from individual PipelineExecutor.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .pipeline_executor import PipelineExecutor

logger = logging.getLogger(__name__)


class BatchPipelineExecutor:
    """Batch executor for processing multiple series in parallel (CRÍTICO-2).
    
    Processes series in batches using ThreadPoolExecutor to control memory usage.
    Individual engine failures do not stop the batch.
    
    Attributes:
        _executor: PipelineExecutor instance for individual series.
        _max_workers: Maximum parallel workers.
        _batch_size: Number of series per batch.
    
    Applies SRP: Only orchestrates parallel execution, delegates to PipelineExecutor.
    """
    
    def __init__(
        self,
        executor: "PipelineExecutor",
        max_workers: int = 8,
        batch_size: int = 50,
    ) -> None:
        """Initialize batch executor.
        
        Args:
            executor: PipelineExecutor instance for individual series.
            max_workers: Maximum parallel workers (default: 8).
            batch_size: Series per batch to control memory (default: 50).
        """
        if max_workers <= 0:
            raise ValueError(f"max_workers must be > 0, got {max_workers}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        
        self._executor = executor
        self._max_workers = max_workers
        self._batch_size = batch_size
    
    def execute_batch(
        self,
        orchestrator: Any,
        series_batch: List[Tuple[str, List[float], Optional[List[float]]]],
        **kwargs,
    ) -> Dict[str, Any]:
        """Execute pipeline for multiple series in parallel.
        
        Args:
            orchestrator: Orchestrator instance to pass to executor.
            series_batch: List of (series_id, values, timestamps) tuples.
            **kwargs: Additional arguments to pass to executor.
        
        Returns:
            Dict with:
                - results: List of successful results
                - errors: List of (series_id, error) tuples
                - total: Total series processed
                - succeeded: Number of successful executions
                - failed: Number of failed executions
        
        Applies CRÍTICO-2: Individual failures do not stop batch processing.
        """
        results = []
        errors = []
        
        # Process in batches to control memory
        for batch_start in range(0, len(series_batch), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(series_batch))
            current_batch = series_batch[batch_start:batch_end]
            
            logger.info(
                "batch_pipeline_processing",
                extra={
                    "batch_start": batch_start,
                    "batch_end": batch_end,
                    "batch_size": len(current_batch),
                    "total_series": len(series_batch),
                },
            )
            
            # Execute batch in parallel
            batch_results, batch_errors = self._execute_batch_parallel(
                orchestrator,
                current_batch,
                **kwargs,
            )
            
            results.extend(batch_results)
            errors.extend(batch_errors)
        
        return {
            "results": results,
            "errors": errors,
            "total": len(series_batch),
            "succeeded": len(results),
            "failed": len(errors),
        }
    
    def _execute_batch_parallel(
        self,
        orchestrator: Any,
        batch: List[Tuple[str, List[float], Optional[List[float]]]],
        **kwargs,
    ) -> Tuple[List[Any], List[Tuple[str, str]]]:
        """Execute batch in parallel using ThreadPoolExecutor.
        
        Args:
            orchestrator: Orchestrator instance.
            batch: List of (series_id, values, timestamps) tuples.
            **kwargs: Additional arguments.
        
        Returns:
            Tuple of (successful_results, errors).
        
        Thread-safe: Each future executes independently.
        """
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            # Submit all tasks
            future_to_series = {}
            for series_id, values, timestamps in batch:
                future = pool.submit(
                    self._execute_single_safe,
                    orchestrator,
                    series_id,
                    values,
                    timestamps,
                    **kwargs,
                )
                future_to_series[future] = series_id
            
            # Collect results as they complete
            for future in as_completed(future_to_series):
                series_id = future_to_series[future]
                try:
                    result = future.result()
                    if result is not None:
                        results.append(result)
                except Exception as exc:
                    # CRÍTICO-2: Individual failure does not stop batch
                    error_msg = str(exc)
                    errors.append((series_id, error_msg))
                    logger.warning(
                        "batch_pipeline_series_failed",
                        extra={
                            "series_id": series_id,
                            "error": error_msg[:200],  # Truncate long errors
                        },
                    )
        
        return results, errors
    
    def _execute_single_safe(
        self,
        orchestrator: Any,
        series_id: str,
        values: List[float],
        timestamps: Optional[List[float]],
        **kwargs,
    ) -> Optional[Any]:
        """Execute pipeline for single series with exception handling.
        
        Args:
            orchestrator: Orchestrator instance.
            series_id: Series identifier.
            values: Sensor values.
            timestamps: Optional timestamps.
            **kwargs: Additional arguments.
        
        Returns:
            Pipeline result or None if execution failed.
        
        Applies CRÍTICO-2: Captures exceptions to prevent thread pool shutdown.
        """
        try:
            return self._executor.execute(
                orchestrator=orchestrator,
                values=values,
                timestamps=timestamps,
                series_id=series_id,
                **kwargs,
            )
        except Exception as exc:
            # Log and re-raise for future.result() to catch
            logger.error(
                "batch_pipeline_execution_error",
                extra={
                    "series_id": series_id,
                    "error": str(exc)[:200],
                },
            )
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get batch executor metrics.
        
        Returns:
            Dict with executor configuration.
        """
        return {
            "max_workers": self._max_workers,
            "batch_size": self._batch_size,
        }
