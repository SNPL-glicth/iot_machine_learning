"""Structured logging module for ML service.

Provides JSON-formatted logs with context for observability.
"""

from .structured_logger import (
    get_logger,
    configure_logging,
    LogContext,
)

__all__ = [
    "get_logger",
    "configure_logging",
    "LogContext",
]
