"""Action guard — validates actions against series operational state.

Ensures recommended actions are coherent with the current state of the
series. Prevents false positives like triggering critical actions on
offline or initializing series.

Pure domain logic — no I/O, no external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class GuardedAction:
    """Action after state-based validation.

    Attributes:
        action_allowed: True if action can proceed, False if suppressed.
        original_action: The original recommended action (if any).
        final_action: The action to use (None if suppressed, may be degraded).
        suppressed_reason: Why action was suppressed (None if allowed).
        series_state: The series state that was evaluated.
    """

    action_allowed: bool
    original_action: Optional[str]
    final_action: Optional[str]
    suppressed_reason: Optional[str]
    series_state: str


class ActionGuard:
    """Guards actions based on series operational state.

    Stateless domain service — validates whether a recommended action
    should be allowed given the current state of the series.
    """

    # States that suppress all actions
    SUPPRESS_ALL_STATES: set[str] = {"INITIALIZING", "OFFLINE"}

    # States that suppress non-critical actions
    STALE_STATE: str = "STALE"

    # Unknown state handling
    UNKNOWN_STATE: str = "UNKNOWN"

    # Active states that allow actions
    ACTIVE_STATES: set[str] = {"ACTIVE", "ONLINE", "RUNNING"}

    # Severity thresholds
    CRITICAL_SEVERITY: str = "CRITICAL"

    def guard(
        self,
        action_required: bool,
        recommended_action: Optional[str],
        severity: str,
        series_state: str,
    ) -> GuardedAction:
        """Validate action against series state.

        Suppression rules:
        - INITIALIZING → suppress all actions, reason="series_not_ready"
        - OFFLINE → suppress all actions, reason="series_offline"
        - STALE → suppress if severity < CRITICAL, reason="series_stale"
        - UNKNOWN → degrade to WARNING max, reason="series_state_unknown"
        - ACTIVE/ONLINE/RUNNING → pass action unchanged
        - Unrecognized state → treated as UNKNOWN

        Args:
            action_required: Whether any action was recommended.
            recommended_action: The action string (if action_required).
            severity: Severity level from classification.
            series_state: Current operational state of the series.

        Returns:
            GuardedAction with validation result.
        """
        # Normalize state (uppercase, strip)
        normalized_state = (series_state or "UNKNOWN").upper().strip()

        # Handle unrecognized states as UNKNOWN
        recognized_states = (
            self.SUPPRESS_ALL_STATES
            | {self.STALE_STATE, self.UNKNOWN_STATE}
            | self.ACTIVE_STATES
        )
        if normalized_state not in recognized_states:
            normalized_state = self.UNKNOWN_STATE

        # No action required → pass through without modification
        if not action_required or not recommended_action:
            return GuardedAction(
                action_allowed=True,
                original_action=recommended_action,
                final_action=recommended_action,
                suppressed_reason=None,
                series_state=normalized_state,
            )

        # Rule 1: INITIALIZING → suppress all
        if normalized_state == "INITIALIZING":
            return GuardedAction(
                action_allowed=False,
                original_action=recommended_action,
                final_action=None,
                suppressed_reason="series_not_ready",
                series_state=normalized_state,
            )

        # Rule 2: OFFLINE → suppress all
        if normalized_state == "OFFLINE":
            return GuardedAction(
                action_allowed=False,
                original_action=recommended_action,
                final_action=None,
                suppressed_reason="series_offline",
                series_state=normalized_state,
            )

        # Rule 3: STALE → suppress if not CRITICAL
        if normalized_state == self.STALE_STATE:
            if severity != self.CRITICAL_SEVERITY:
                return GuardedAction(
                    action_allowed=False,
                    original_action=recommended_action,
                    final_action=None,
                    suppressed_reason="series_stale",
                    series_state=normalized_state,
                )
            # CRITICAL severity passes through even if STALE
            return GuardedAction(
                action_allowed=True,
                original_action=recommended_action,
                final_action=recommended_action,
                suppressed_reason=None,
                series_state=normalized_state,
            )

        # Rule 4: UNKNOWN → degrade severity to WARNING max
        if normalized_state == self.UNKNOWN_STATE:
            # For unknown state, we allow the action but note the degradation
            # The caller should handle degrading WARNING to something safer
            return GuardedAction(
                action_allowed=True,
                original_action=recommended_action,
                final_action=recommended_action,  # Caller may further degrade
                suppressed_reason="series_state_unknown",
                series_state=normalized_state,
            )

        # Rule 5: ACTIVE/ONLINE/RUNNING → pass unchanged
        if normalized_state in self.ACTIVE_STATES:
            return GuardedAction(
                action_allowed=True,
                original_action=recommended_action,
                final_action=recommended_action,
                suppressed_reason=None,
                series_state=normalized_state,
            )

        # Fallback (should not reach here due to unrecognized→UNKNOWN above)
        return GuardedAction(
            action_allowed=True,
            original_action=recommended_action,
            final_action=recommended_action,
            suppressed_reason=None,
            series_state=normalized_state,
        )
