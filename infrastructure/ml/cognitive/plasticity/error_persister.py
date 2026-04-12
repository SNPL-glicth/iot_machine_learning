"""Error persistence helper for advanced plasticity.

Isolated storage persistence logic.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


class ErrorPersister:
    """Handles persistence of errors to storage."""
    
    def __init__(self, storage_adapter: Optional[Any] = None) -> None:
        self._storage = storage_adapter
    
    def persist_contextual_error(
        self,
        series_id: str,
        engine_name: str,
        predicted_value: float,
        actual_value: float,
        error: float,
        penalty: float,
        plasticity_context,
    ) -> None:
        """Persist error to storage if available."""
        if not self._storage or not hasattr(self._storage, 'record_contextual_error'):
            return
        
        try:
            self._storage.record_contextual_error(
                series_id=series_id,
                engine_name=engine_name,
                predicted_value=predicted_value,
                actual_value=actual_value,
                error=error,
                penalty=penalty,
                regime=plasticity_context.regime.value,
                noise_ratio=plasticity_context.noise_ratio,
                volatility=plasticity_context.volatility,
                time_of_day=plasticity_context.time_of_day,
                consecutive_failures=plasticity_context.consecutive_failures,
                is_critical_zone=plasticity_context.is_critical_zone,
                context_key=getattr(plasticity_context, 'context_key', ''),
            )
        except Exception as e:
            logger.error("persist_error_failed", extra={"series_id": series_id, "error": str(e)})
    
    def persist_engine_health(
        self,
        series_id: str,
        engine_name: str,
        health_state,
    ) -> None:
        """Persist engine health to storage if available."""
        if not self._storage or not hasattr(self._storage, 'update_engine_health'):
            return
        
        try:
            self._storage.update_engine_health(
                series_id=series_id,
                engine_name=engine_name,
                consecutive_failures=health_state.consecutive_failures,
                consecutive_successes=health_state.consecutive_successes,
                total_predictions=health_state.total_predictions,
                total_errors=health_state.total_errors,
                last_error=health_state.last_error,
                failure_rate=health_state.failure_rate,
                is_inhibited=health_state.is_inhibited,
                inhibition_reason=health_state.inhibition_reason,
                last_success_time=health_state.last_success_time.isoformat() if health_state.last_success_time else None,
                last_failure_time=health_state.last_failure_time.isoformat() if health_state.last_failure_time else None,
            )
        except Exception as e:
            logger.error("persist_health_failed", extra={"series_id": series_id, "error": str(e)})
