"""Batch writable interface for series stores (PERF-SEV-1).

Applies ISP: Separates batch operations from single-value operations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class IBatchWritable(ABC):
    """Interface for batch write operations (PERF-SEV-1).
    
    Allows stores to implement efficient batch operations.
    
    Applies ISP: Clients that only need batch writes depend on this,
    not on the full store interface.
    """
    
    @abstractmethod
    def append_batch(self, series_id: str, values: List[float]) -> None:
        """Append multiple values in a single operation.
        
        Args:
            series_id: Series identifier.
            values: List of values to append.
        
        Implementation should use Redis pipeline or equivalent
        to minimize round-trips.
        """
        pass
    
    @abstractmethod
    def append_batch_multi_series(
        self,
        batch: List[tuple[str, List[float]]],
    ) -> None:
        """Append values to multiple series in a single operation.
        
        Args:
            batch: List of (series_id, values) tuples.
        
        Implementation should use Redis pipeline to batch all
        operations across multiple series.
        """
        pass
