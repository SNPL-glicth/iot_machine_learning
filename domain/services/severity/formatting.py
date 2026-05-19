"""Pure domain formatting utilities.

No infrastructure imports allowed — string formatting only.
"""

from __future__ import annotations


def format_universal_conclusion(domain: str, severity: str, confidence: float) -> str:
    """Format a simple conclusion string from domain fields.

    Args:
        domain: Domain name (e.g. 'iot', 'security').
        severity: Severity level (e.g. 'critical', 'high').
        confidence: Confidence score 0.0–1.0.

    Returns:
        Formatted conclusion string.
    """
    return f"{domain.title()} incident — {severity.title()} | Confidence: {confidence:.1%}"
