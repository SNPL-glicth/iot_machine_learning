"""Smart inhibition rules — R-2 Refactor.

Implements intelligent inhibition that considers signal context.
During high z-score events (anomalies), prediction errors are expected
and should not trigger inhibition. This rewards engines that detect
anomalies rather than penalizing them.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# R-2: Smart Inhibition threshold
# If signal z-score > this value, reduce inhibition by 80%
_ANOMALY_Z_SCORE_THRESHOLD: float = 2.5
_SMART_INHIBITION_REDUCTION_FACTOR: float = 0.2  # 80% reduction = 20% effective


def check_failure_threshold(
    state,
    failure_threshold: int,
    signal_z_score: Optional[float] = None,
) -> Optional[str]:
    """Return inhibition reason if consecutive failures exceed threshold.

    R-2 Smart Inhibition: If signal_z_score indicates an anomaly,
    the effective threshold is raised by 80%, preventing false
    inhibition during legitimate high-volatility periods.

    Args:
        state: EnginePlasticityState instance.
        failure_threshold: Max allowed consecutive failures.
        signal_z_score: Optional z-score of the signal. If > 2.5,
            the effective threshold is raised (harder to inhibit).

    Returns:
        Inhibition reason string, or None if threshold not exceeded.
    """
    # R-2: Calculate effective threshold considering signal context
    effective_threshold = failure_threshold
    is_anomaly_context = False
    
    if signal_z_score is not None and signal_z_score > _ANOMALY_Z_SCORE_THRESHOLD:
        # During anomalies, require 5x more failures to inhibit
        effective_threshold = int(failure_threshold / _SMART_INHIBITION_REDUCTION_FACTOR)
        is_anomaly_context = True
    
    if state.consecutive_failures >= effective_threshold:
        reason = (
            f"Consecutive failures: {state.consecutive_failures} "
            f">= {effective_threshold}"
        )
        if is_anomaly_context:
            reason += f" (raised from {failure_threshold} due to anomaly context)"
            
        logger.warning(
            "engine_inhibited_failures",
            extra={
                "series_id": state.series_id,
                "engine_name": state.engine_name,
                "consecutive_failures": state.consecutive_failures,
                "threshold": failure_threshold,
                "effective_threshold": effective_threshold,
                "signal_z_score": signal_z_score,
                "smart_inhibition_applied": is_anomaly_context,
            },
        )
        return reason
    return None


def check_timeout(
    state,
    max_hours_without_success: float,
) -> Optional[str]:
    """Return inhibition reason if engine has not succeeded within time limit.

    Args:
        state: EnginePlasticityState instance.
        max_hours_without_success: Maximum allowed hours without a success.

    Returns:
        Inhibition reason string, or None if within time limit.
    """
    hours = state.hours_since_last_success
    if hours is not None and hours > max_hours_without_success:
        reason = (
            f"No success for {hours:.2f} hours "
            f"(max: {max_hours_without_success})"
        )
        logger.warning(
            "engine_inhibited_timeout",
            extra={
                "series_id": state.series_id,
                "engine_name": state.engine_name,
                "hours_since_success": hours,
                "max_hours": max_hours_without_success,
            },
        )
        return reason
    return None


def evaluate_inhibition(
    state,
    failure_threshold: int,
    max_hours_without_success: float,
    signal_z_score: Optional[float] = None,
):
    """Apply all inhibition rules to a state and return updated state.

    R-2: Now considers signal context for smart inhibition.
    
    Checks failure threshold (with smart context), then timeout.
    Returns the state unchanged if already inhibited or no rule triggers.

    Args:
        state: EnginePlasticityState instance.
        failure_threshold: Max allowed consecutive failures.
        max_hours_without_success: Maximum allowed hours without a success.
        signal_z_score: Optional signal context for smart inhibition.

    Returns:
        Possibly updated EnginePlasticityState.
    """
    if state.is_inhibited:
        return state

    # R-2: Pass signal context for smart inhibition
    reason = check_failure_threshold(state, failure_threshold, signal_z_score)
    if reason is None:
        reason = check_timeout(state, max_hours_without_success)

    if reason is not None:
        return state.with_inhibition(reason)

    return state


def build_health_summary(states: dict) -> dict:
    """Build a health summary dict from a series's engine state map.

    Args:
        states: Dict mapping engine_name → EnginePlasticityState.

    Returns:
        Dict mapping engine_name → health metrics dict.
    """
    return {
        engine_name: {
            "consecutive_failures": state.consecutive_failures,
            "consecutive_successes": state.consecutive_successes,
            "total_predictions": state.total_predictions,
            "failure_rate": state.failure_rate,
            "is_inhibited": state.is_inhibited,
            "inhibition_reason": state.inhibition_reason,
            "last_error": state.last_error,
        }
        for engine_name, state in states.items()
    }
