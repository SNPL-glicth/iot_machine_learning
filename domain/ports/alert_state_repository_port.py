"""AlertStateRepositoryPort protocol for alert suppression state persistence."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class AlertStateRepositoryPort(Protocol):
    """Protocol for storing and retrieving alert suppression state."""

    def get_last_alert(self, series_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve last emitted alert for series_id."""
        ...

    def save_alert(
        self,
        series_id: str,
        action: str,
        priority: int,
        severity: str,
    ) -> None:
        """Save emitted alert."""
        ...

    def increment_suppressed(self, series_id: str) -> int:
        """Increment suppressed alerts counter and return new count."""
        ...

    def get_suppressed_count(self, series_id: str) -> int:
        """Get suppressed alerts counter for series_id."""
        ...


class InMemoryAlertStateRepository(AlertStateRepositoryPort):
    """In-memory implementation of AlertStateRepositoryPort."""

    def __init__(self) -> None:
        self._last_alerts: Dict[str, Dict[str, Any]] = {}
        self._suppressed_counts: Dict[str, int] = {}

    def get_last_alert(self, series_id: str) -> Optional[Dict[str, Any]]:
        return self._last_alerts.get(series_id)

    def save_alert(
        self,
        series_id: str,
        action: str,
        priority: int,
        severity: str,
    ) -> None:
        import time
        self._last_alerts[series_id] = {
            "action": action,
            "priority": priority,
            "timestamp": time.time(),
            "severity": severity,
        }

    def increment_suppressed(self, series_id: str) -> int:
        count = self._suppressed_counts.get(series_id, 0) + 1
        self._suppressed_counts[series_id] = count
        return count

    def get_suppressed_count(self, series_id: str) -> int:
        return self._suppressed_counts.get(series_id, 0)
