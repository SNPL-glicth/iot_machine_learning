"""Priority constants and mappings for decision engine.

Domain-pure constants for priority levels and severity-to-priority mappings.
"""

from __future__ import annotations

from typing import Dict


class Priority:
    """Priority level constants for decisions."""

    CRITICAL = 1  # Immediate action required
    HIGH = 2      # Action required soon
    MEDIUM = 3    # Monitor closely
    LOW = 4       # Normal monitoring


# Severity to priority mapping
SEVERITY_PRIORITY_MAP: Dict[str, int] = {
    "critical": Priority.CRITICAL,
    "warning": Priority.HIGH,
    "info": Priority.LOW,
}

# Priority to action mapping
PRIORITY_ACTION_MAP: Dict[int, str] = {
    Priority.CRITICAL: "escalate",
    Priority.HIGH: "investigate",
    Priority.MEDIUM: "monitor",
    Priority.LOW: "monitor",
}

# Priority labels for human readability
PRIORITY_LABELS: Dict[int, str] = {
    Priority.CRITICAL: "critical",
    Priority.HIGH: "high",
    Priority.MEDIUM: "medium",
    Priority.LOW: "low",
}
