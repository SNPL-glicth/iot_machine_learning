"""Inhibition rules for EngineHealthMonitor.

Pure functions that decide whether an engine should be inhibited
based on its current state. No state, no I/O, no threading.

Extracted from EngineHealthMonitor to keep that class focused on
state tracking and thread-safe access only.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def check_failure_threshold(
    state,
    failure_threshold: int,
) -> Optional[str]:
    """Return inhibition reason if consecutive failures exceed threshold.

    Args:
        state: EnginePlasticityState instance.
        failure_threshold: Max allowed consecutive failures.

    Returns:
        Inhibition reason string, or None if threshold not exceeded.
    """
    if state.consecutive_failures >= failure_threshold:
        reason = (
            f"Consecutive failures: {state.consecutive_failures} "
            f">= {failure_threshold}"
        )
        logger.warning(
            "engine_inhibited_failures",
            extra={
                "series_id": state.series_id,
                "engine_name": state.engine_name,
                "consecutive_failures": state.consecutive_failures,
                "threshold": failure_threshold,
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
):
    """Apply all inhibition rules to a state and return updated state.

    Checks failure threshold first, then timeout. Returns the state
    unchanged if already inhibited or no rule triggers.

    Args:
        state: EnginePlasticityState instance.
        failure_threshold: Max allowed consecutive failures.
        max_hours_without_success: Maximum allowed hours without a success.

    Returns:
        Possibly updated EnginePlasticityState.
    """
    if state.is_inhibited:
        return state

    reason = check_failure_threshold(state, failure_threshold)
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
