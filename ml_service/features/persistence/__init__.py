"""Persistencia de ventanas ML en Redis.

FIX 2026-02-02: Implementa persistencia para evitar pérdida de contexto (ML-2).
"""

from .redis_window_store import RedisWindowStore, get_window_store

__all__ = ["RedisWindowStore", "get_window_store"]
