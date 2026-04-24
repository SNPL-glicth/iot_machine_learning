"""Handler for recording actual values in the cognitive orchestrator.

Extracted from MetaCognitiveOrchestrator.record_actual to keep the
orchestrator under the 300-line complexity guard.

Two paths:
- Advanced plasticity: delegates to AdvancedPlasticityCoordinator
- Legacy plasticity: direct error tracking + PlasticityTracker update
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def record_actual_legacy(
    actual_value: float,
    perceptions: List,
    regime: str,
    error_history,
    plasticity_tracker,
    storage,
    series_id: Optional[str],
    reliability_tracker=None,
) -> None:
    """Direct error tracking path.

    Args:
        actual_value: True observed value.
        perceptions: List of EnginePerception from last predict().
        regime: Signal regime string from last predict().
        error_history: ErrorHistoryManager instance (CRIT-1 fix: series-scoped).
        plasticity_tracker: Optional BayesianWeightTracker instance.
        storage: Optional storage adapter for recording errors.
        series_id: Optional series identifier for storage.
        reliability_tracker: Optional :class:`EngineReliabilityTracker`
            (IMP-4b). When provided, each outcome updates the Beta
            posterior so :meth:`is_reliable` reflects the latest
            evidence by the next ``predict()``.
    """
    for p in perceptions:
        error = abs(p.predicted_value - actual_value)
        # CRIT-1: Record error with series_id namespace isolation
        if error_history is not None and series_id is not None:
            error_history.record_error(series_id, p.engine_name, error)

        if plasticity_tracker is not None:
            plasticity_tracker.update(
                regime, p.engine_name, error, series_id=series_id
            )

        if reliability_tracker is not None and series_id is not None:
            reliability_tracker.record_outcome(series_id, p.engine_name, error)

        if storage and series_id:
            storage.record_prediction_error(
                series_id=series_id,
                engine_name=p.engine_name,
                predicted_value=p.predicted_value,
                actual_value=actual_value,
            )


def record_actual_dispatch(
    actual_value: float,
    last_regime: Optional[str],
    last_perceptions: List,
    last_signal_context,
    enable_advanced_plasticity: bool,
    plasticity_coordinator,
    plasticity_tracker,
    error_history,
    storage,
    series_id: Optional[str],
    series_context,
    reliability_tracker=None,
) -> None:
    """Dispatch record_actual to the appropriate plasticity path.

    Returns immediately if there is no regime or no perceptions recorded
    (i.e. predict() has not been called yet).

    Args:
        actual_value: True observed value.
        last_regime: Regime string from last predict() call.
        last_perceptions: Perceptions from last predict() call.
        last_signal_context: SignalContext from last predict() call.
        enable_advanced_plasticity: Whether advanced plasticity is active.
        plasticity_coordinator: AdvancedPlasticityCoordinator or None.
        plasticity_tracker: PlasticityTracker or None.
        error_history: ErrorHistoryManager instance (CRIT-1 fix).
        storage: Optional storage adapter.
        series_id: Optional series identifier.
        series_context: Optional SeriesContext for asymmetric penalty.
    """
    if last_regime is None or not last_perceptions:
        return

    if (
        enable_advanced_plasticity
        and last_signal_context is not None
        and plasticity_coordinator is not None
    ):
        plasticity_coordinator.record_actual_advanced(
            actual_value=actual_value,
            perceptions=last_perceptions,
            plasticity_context=last_signal_context,
            regime=last_regime,
            series_id=series_id,
            series_context=series_context,
            plasticity_tracker=plasticity_tracker,
            error_history=error_history,
        )
    else:
        record_actual_legacy(
            actual_value=actual_value,
            perceptions=last_perceptions,
            regime=last_regime,
            error_history=error_history,
            plasticity_tracker=plasticity_tracker,
            storage=storage,
            series_id=series_id,
            reliability_tracker=reliability_tracker,
        )
