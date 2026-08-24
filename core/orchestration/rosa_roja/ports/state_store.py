"""Port for ML state persistence backends.

Infrastructure implements this port (Redis, in-memory, file). The engine
depends only on this protocol; a None store disables persistence entirely
(zero behavioral change).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, runtime_checkable


@runtime_checkable
class MLStateStore(Protocol):
    """Key-value snapshot store for engine learning state."""

    def save(self, engine_id: str, payload: Dict[str, Any]) -> bool:
        """Persist an atomic engine snapshot. Returns True on success."""
        ...

    def load(self, engine_id: str) -> Optional[Dict[str, Any]]:
        """Load the latest snapshot, or None if absent/unreadable."""
        ...

    def delete(self, engine_id: str) -> bool:
        """Remove the snapshot. Returns True if deleted."""
        ...
