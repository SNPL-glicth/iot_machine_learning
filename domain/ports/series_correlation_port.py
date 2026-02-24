from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class SeriesCorrelationPort(ABC):
    """Port for accessing correlation information between series.
    
    Enables the cognitive orchestrator to enrich predictions with
    spatial/temporal context from correlated series.
    """
    
    @abstractmethod
    def get_correlated_series(
        self,
        series_id: str,
        max_neighbors: int = 5,
    ) -> List[Tuple[str, float]]:
        """Get correlated series for a given series.
        
        Args:
            series_id: Series identifier
            max_neighbors: Maximum number of neighbors to return
        
        Returns:
            List of (neighbor_series_id, correlation_coefficient) tuples,
            sorted by correlation strength (highest first)
        """
        pass
    
    @abstractmethod
    def get_recent_values_multi(
        self,
        series_ids: List[str],
        window: int,
    ) -> Dict[str, List[float]]:
        """Load recent values for multiple series in a single query.
        
        Args:
            series_ids: List of series identifiers
            window: Number of recent values to load per series
        
        Returns:
            Dict mapping series_id to list of recent values (chronological order)
        """
        pass
