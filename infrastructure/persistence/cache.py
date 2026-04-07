"""Cache en memoria para análisis de documentos."""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

from iot_machine_learning.domain.ports.document_analysis import (
    AnalysisOutput,
    CachePort,
)

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Entrada de cache con TTL."""
    value: AnalysisOutput
    expires_at: float
    created_at: float


class InMemoryAnalysisCache(CachePort):
    """Cache en memoria con TTL y LRU eviction.
    
    Thread-safe. Reemplazable por Redis en producción.
    
    Args:
        ttl_seconds: Tiempo de vida por defecto
        max_size: Máximo número de entradas
    """
    
    def __init__(
        self,
        ttl_seconds: int = 300,
        max_size: int = 500,
    ) -> None:
        """Inicializa cache."""
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[AnalysisOutput]:
        """Obtiene resultado cacheado si no expiró."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            
            # Check TTL
            now = time.time()
            if now > entry.expires_at:
                del self._store[key]
                logger.debug(f"cache_expired: key={key[:20]}...")
                return None
            
            # Move to end (LRU)
            self._store.move_to_end(key)
            logger.debug(f"cache_hit: key={key[:20]}...")
            return entry.value
    
    def set(
        self,
        key: str,
        value: AnalysisOutput,
        ttl: Optional[int] = None,
    ) -> None:
        """Guarda resultado en cache con TTL."""
        with self._lock:
            # Evict if at capacity
            if len(self._store) >= self._max_size:
                self._evict_lru()
            
            # Evict expired entries
            self._evict_expired()
            
            # Store new entry
            now = time.time()
            ttl_seconds = ttl if ttl is not None else self._ttl
            entry = CacheEntry(
                value=value,
                expires_at=now + ttl_seconds,
                created_at=now,
            )
            self._store[key] = entry
            
            # Move to end (most recent)
            self._store.move_to_end(key)
            
            logger.debug(
                f"cache_stored: key={key[:20]}..., "
                f"size={len(self._store)}, ttl={ttl_seconds}s"
            )
    
    def invalidate(self, key: str) -> None:
        """Invalida entrada de cache."""
        with self._lock:
            if key in self._store:
                del self._store[key]
                logger.debug(f"cache_invalidated: key={key[:20]}...")
    
    def _evict_lru(self) -> None:
        """Evict least recently used entries (10% of capacity)."""
        n_to_evict = max(1, self._max_size // 10)
        for _ in range(n_to_evict):
            if self._store:
                key, _ = self._store.popitem(last=False)
                logger.debug(f"cache_evicted_lru: key={key[:20]}...")
    
    def _evict_expired(self) -> None:
        """Evict expired entries."""
        now = time.time()
        expired_keys = [
            key for key, entry in self._store.items()
            if now > entry.expires_at
        ]
        for key in expired_keys:
            del self._store[key]
        
        if expired_keys:
            logger.debug(f"cache_evicted_expired: count={len(expired_keys)}")
    
    @property
    def size(self) -> int:
        """Tamaño actual del cache."""
        with self._lock:
            return len(self._store)
    
    def clear(self) -> None:
        """Limpia todo el cache."""
        with self._lock:
            self._store.clear()
            logger.info("cache_cleared")


def compute_content_hash(content: str) -> str:
    """Computa hash MD5 del contenido."""
    return hashlib.md5(content.encode('utf-8')).hexdigest()[:16]


def build_cache_key(content_hash: str, content_type: str) -> str:
    """Construye key de cache."""
    return f"{content_hash}:{content_type}"
