"""Redis configuration and constants.

Extracted from redis_connection_manager.py as part of modularization.
"""

from __future__ import annotations

# Connection defaults
DEFAULT_REDIS_URL = "redis://localhost:6379/0"
DEFAULT_SOCKET_TIMEOUT = 5.0
DEFAULT_SOCKET_CONNECT_TIMEOUT = 5.0

# Pool sizing — separate pools for different use cases
GENERAL_MAX_CONNECTIONS = 150  # For cache, ML, API (fast operations)
STREAM_MAX_CONNECTIONS = 10    # For blocking stream operations
DEFAULT_MAX_CONNECTIONS = 50   # Legacy compatibility
