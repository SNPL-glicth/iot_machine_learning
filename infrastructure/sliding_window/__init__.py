"""Canonical sliding window implementations (E-15).

Provides ISlidingWindowStore implementations:
- InMemorySlidingWindowStore: LRU+TTL per-sensor store (canonical)
"""

from .in_memory import InMemorySlidingWindowStore

__all__ = ["InMemorySlidingWindowStore"]
