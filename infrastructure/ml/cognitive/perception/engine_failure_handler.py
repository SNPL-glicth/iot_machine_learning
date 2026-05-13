"""Engine failure handler for parallel execution (PERF-CRIT-1).

Handles individual engine failures without cascading to others.

Applies SRP: Failure handling is separate concern from execution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class EngineFailure:
    """Record of an engine failure.
    
    Attributes:
        engine_name: Name of failed engine.
        error_message: Error description.
        error_type: Exception type name.
        timestamp: Failure timestamp (monotonic).
    """
    engine_name: str
    error_message: str
    error_type: str
    timestamp: float


class EngineFailureHandler:
    """Handles individual engine failures (PERF-CRIT-1).
    
    Isolates failures to prevent cascading to other engines.
    
    Applies SRP: Only handles failures, doesn't execute engines.
    """
    
    def __init__(self, max_failures_to_track: int = 100) -> None:
        """Initialize failure handler.
        
        Args:
            max_failures_to_track: Maximum failures to keep in memory.
        """
        self._failures: list[EngineFailure] = []
        self._max_failures = max_failures_to_track
    
    def record_failure(
        self,
        engine_name: str,
        exception: Exception,
        timestamp: float,
    ) -> None:
        """Record an engine failure.
        
        Args:
            engine_name: Name of failed engine.
            exception: Exception that occurred.
            timestamp: Failure timestamp.
        """
        failure = EngineFailure(
            engine_name=engine_name,
            error_message=str(exception),
            error_type=type(exception).__name__,
            timestamp=timestamp,
        )
        
        self._failures.append(failure)
        
        # Trim if too many
        if len(self._failures) > self._max_failures:
            self._failures = self._failures[-self._max_failures:]
        
        logger.warning(
            f"Engine failure recorded: {engine_name}",
            extra={
                "engine_name": engine_name,
                "error_type": failure.error_type,
                "error_message": failure.error_message,
            }
        )
    
    def get_recent_failures(
        self,
        engine_name: Optional[str] = None,
        limit: int = 10,
    ) -> list[EngineFailure]:
        """Get recent failures.
        
        Args:
            engine_name: Filter by engine name (None = all).
            limit: Maximum failures to return.
        
        Returns:
            List of recent failures, newest first.
        """
        failures = self._failures
        
        if engine_name is not None:
            failures = [f for f in failures if f.engine_name == engine_name]
        
        return list(reversed(failures[-limit:]))
    
    def get_failure_count(self, engine_name: Optional[str] = None) -> int:
        """Get failure count.
        
        Args:
            engine_name: Filter by engine name (None = all).
        
        Returns:
            Number of failures.
        """
        if engine_name is None:
            return len(self._failures)
        
        return sum(1 for f in self._failures if f.engine_name == engine_name)
    
    def clear_failures(self, engine_name: Optional[str] = None) -> None:
        """Clear failure records.
        
        Args:
            engine_name: Clear only for this engine (None = all).
        """
        if engine_name is None:
            self._failures.clear()
        else:
            self._failures = [
                f for f in self._failures
                if f.engine_name != engine_name
            ]
