"""Tests for ActionGuard domain service.

Covers state-based action suppression rules:
- INITIALIZING → suppress all actions
- OFFLINE → suppress all actions
- STALE → suppress if severity < CRITICAL
- UNKNOWN → degrade to WARNING
- ACTIVE/ONLINE → pass unchanged
- Unrecognized states → treated as UNKNOWN
"""

from __future__ import annotations

import pytest

from iot_machine_learning.domain.services.action_guard import (
    ActionGuard,
    GuardedAction,
)


class TestActionGuardBasics:
    """Basic construction and sanity checks."""

    def test_guard_construction(self):
        """ActionGuard can be instantiated."""
        guard = ActionGuard()
        assert guard is not None

    def test_guarded_action_dataclass(self):
        """GuardedAction is a proper frozen dataclass."""
        result = GuardedAction(
            action_allowed=False,
            original_action="alert_critical",
            final_action=None,
            suppressed_reason="series_offline",
            series_state="OFFLINE",
        )
        assert result.action_allowed is False
        assert result.original_action == "alert_critical"
        assert result.final_action is None
        assert result.suppressed_reason == "series_offline"
        assert result.series_state == "OFFLINE"


class TestNoActionRequired:
    """When no action required, pass through unchanged."""

    def test_no_action_required(self):
        """action_required=False → pass through unchanged."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=False,
            recommended_action=None,
            severity="NORMAL",
            series_state="ONLINE",
        )
        assert result.action_allowed is True
        assert result.original_action is None
        assert result.final_action is None
        assert result.suppressed_reason is None

    def test_no_recommended_action(self):
        """Empty recommended_action → pass through."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action=None,
            severity="WARNING",
            series_state="ONLINE",
        )
        assert result.action_allowed is True
        assert result.final_action is None


class TestInitializingState:
    """INITIALIZING state suppresses all actions."""

    def test_initializing_suppresses_critical(self):
        """INITIALIZING + CRITICAL → suppressed."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert_critical",
            severity="CRITICAL",
            series_state="INITIALIZING",
        )
        assert result.action_allowed is False
        assert result.final_action is None
        assert result.suppressed_reason == "series_not_ready"
        assert result.series_state == "INITIALIZING"

    def test_initializing_suppresses_warning(self):
        """INITIALIZING + WARNING → suppressed."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert_warning",
            severity="WARNING",
            series_state="INITIALIZING",
        )
        assert result.action_allowed is False
        assert result.suppressed_reason == "series_not_ready"

    def test_initializing_case_insensitive(self):
        """Initializing (lowercase) → treated as INITIALIZING."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert",
            severity="CRITICAL",
            series_state="initializing",
        )
        assert result.action_allowed is False
        assert result.suppressed_reason == "series_not_ready"


class TestOfflineState:
    """OFFLINE state suppresses all actions."""

    def test_offline_suppresses_all(self):
        """OFFLINE + any severity → suppressed."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert_critical",
            severity="CRITICAL",
            series_state="OFFLINE",
        )
        assert result.action_allowed is False
        assert result.final_action is None
        assert result.suppressed_reason == "series_offline"

    def test_offline_suppresses_warning(self):
        """OFFLINE + WARNING → suppressed."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert_warning",
            severity="WARNING",
            series_state="OFFLINE",
        )
        assert result.action_allowed is False
        assert result.suppressed_reason == "series_offline"


class TestStaleState:
    """STALE state suppresses non-critical actions."""

    def test_stale_suppresses_warning(self):
        """STALE + WARNING → suppressed."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert_warning",
            severity="WARNING",
            series_state="STALE",
        )
        assert result.action_allowed is False
        assert result.final_action is None
        assert result.suppressed_reason == "series_stale"

    def test_stale_suppresses_normal(self):
        """STALE + NORMAL → suppressed."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="log_event",
            severity="NORMAL",
            series_state="STALE",
        )
        assert result.action_allowed is False

    def test_stale_allows_critical(self):
        """STALE + CRITICAL → allowed (passes through)."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert_critical",
            severity="CRITICAL",
            series_state="STALE",
        )
        assert result.action_allowed is True
        assert result.final_action == "alert_critical"
        assert result.suppressed_reason is None


class TestUnknownState:
    """UNKNOWN state degrades action (allows with warning)."""

    def test_unknown_allows_with_reason(self):
        """UNKNOWN + any severity → allowed but with suppressed_reason."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert_critical",
            severity="CRITICAL",
            series_state="UNKNOWN",
        )
        assert result.action_allowed is True
        assert result.final_action == "alert_critical"
        assert result.suppressed_reason == "series_state_unknown"

    def test_unknown_degrades_warning(self):
        """UNKNOWN + WARNING → allowed but flagged."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert_warning",
            severity="WARNING",
            series_state="UNKNOWN",
        )
        assert result.action_allowed is True
        assert result.suppressed_reason == "series_state_unknown"


class TestActiveStates:
    """ACTIVE/ONLINE/RUNNING states allow actions unchanged."""

    def test_active_allows_action(self):
        """ACTIVE → action passes unchanged."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert_critical",
            severity="CRITICAL",
            series_state="ACTIVE",
        )
        assert result.action_allowed is True
        assert result.final_action == "alert_critical"
        assert result.suppressed_reason is None

    def test_online_allows_action(self):
        """ONLINE → action passes unchanged."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert_warning",
            severity="WARNING",
            series_state="ONLINE",
        )
        assert result.action_allowed is True
        assert result.final_action == "alert_warning"
        assert result.suppressed_reason is None

    def test_running_allows_action(self):
        """RUNNING → action passes unchanged."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="log_event",
            severity="NORMAL",
            series_state="RUNNING",
        )
        assert result.action_allowed is True
        assert result.final_action == "log_event"


class TestUnrecognizedStates:
    """Unrecognized states are treated as UNKNOWN."""

    def test_unrecognized_state_as_unknown(self):
        """WEIRD_STATE → treated as UNKNOWN."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert",
            severity="WARNING",
            series_state="WEIRD_STATE",
        )
        assert result.action_allowed is True  # UNKNOWN behavior
        assert result.suppressed_reason == "series_state_unknown"
        assert result.series_state == "UNKNOWN"

    def test_empty_state_as_unknown(self):
        """Empty string → treated as UNKNOWN."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert",
            severity="CRITICAL",
            series_state="",
        )
        assert result.series_state == "UNKNOWN"
        assert result.suppressed_reason == "series_state_unknown"

    def test_none_state_as_unknown(self):
        """None state → treated as UNKNOWN."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert",
            severity="CRITICAL",
            series_state=None,  # type: ignore
        )
        assert result.series_state == "UNKNOWN"


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_case_insensitive_states(self):
        """State matching is case-insensitive."""
        guard = ActionGuard()
        
        result_lower = guard.guard(
            action_required=True,
            recommended_action="alert",
            severity="CRITICAL",
            series_state="offline",
        )
        assert result_lower.action_allowed is False
        
        result_mixed = guard.guard(
            action_required=True,
            recommended_action="alert",
            severity="CRITICAL",
            series_state="Offline",
        )
        assert result_mixed.action_allowed is False

    def test_whitespace_handling(self):
        """Whitespace is stripped from state."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action="alert",
            severity="CRITICAL",
            series_state="  OFFLINE  ",
        )
        assert result.action_allowed is False
        assert result.series_state == "OFFLINE"

    def test_none_action_with_action_required(self):
        """action_required=True but recommended_action=None."""
        guard = ActionGuard()
        result = guard.guard(
            action_required=True,
            recommended_action=None,
            severity="WARNING",
            series_state="ONLINE",
        )
        assert result.action_allowed is True
        assert result.final_action is None
