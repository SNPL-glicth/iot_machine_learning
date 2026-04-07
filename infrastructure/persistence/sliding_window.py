"""Implementación canónica de sliding window — unifica 3 implementaciones legacy.

Reemplaza:
- SlidingWindowStore (ml_service/consumers/sliding_window.py)
- SlidingWindowBuffer (ml_service/sliding_window_buffer.py)
- InMemorySlidingWindowStore (infrastructure/sliding_window/in_memory.py)

Thread-safe, LRU + TTL eviction, generic sobre tipo de dato.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Deque, Generic, List, Optional, Tuple, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class WindowEntry(Generic[T]):
    """Entrada de ventana con metadatos de acceso."""
    
    window: Deque[Tuple[float, T]]
    created_at: float = field(default_factory=time.monotonic)
    last_access: float = field(default_factory=time.monotonic)


class SlidingWindowCache(Generic[T]):
    """Ventana deslizante canónica. Thread-safe, LRU, TTL opcional.
    
    Características:
    - Thread-safe con RLock (permite reentrada)
    - LRU eviction cuando se alcanza max_series
    - TTL eviction para series inactivas
    - Generic sobre tipo de dato almacenado
    - series_id siempre str (int se convierte en adapters)
    
    Args:
        window_size: Máximo de puntos por ventana
        max_series: Máximo de series simultáneas
        ttl_seconds: Tiempo de vida para series inactivas (None = sin TTL)
    """
    
    def __init__(
        self,
        window_size: int = 20,
        max_series: int = 1000,
        ttl_seconds: Optional[int] = 3600,
    ) -> None:
        """Inicializa cache de ventanas."""
        self._window_size = max(1, window_size)
        self._max_series = max(1, max_series)
        self._ttl = float(ttl_seconds) if ttl_seconds is not None else None
        self._store: OrderedDict[str, WindowEntry[T]] = OrderedDict()
        self._lock = threading.RLock()
        self._evictions_lru: int = 0
        self._evictions_ttl: int = 0
    
    def append(
        self,
        series_id: str,
        value: T,
        timestamp: Optional[float] = None,
    ) -> int:
        """Agrega punto a la ventana de series_id."""
        ts = timestamp if timestamp is not None else time.time()
        now = time.monotonic()
        
        with self._lock:
            # Evict expired first
            if self._ttl is not None:
                self._evict_expired(now)
            
            # Get or create entry
            if series_id in self._store:
                entry = self._store[series_id]
                self._store.move_to_end(series_id)
            else:
                self._evict_lru_if_full()
                entry = WindowEntry(
                    window=deque(maxlen=self._window_size),
                    created_at=now,
                    last_access=now,
                )
                self._store[series_id] = entry
            
            # Append to window
            entry.window.append((ts, value))
            entry.last_access = now
            
            return len(entry.window)
    
    def get(self, series_id: str) -> Optional[List[Tuple[float, T]]]:
        """Retorna lista de (timestamp, value) o None si no existe."""
        with self._lock:
            entry = self._store.get(series_id)
            if entry is None:
                return None
            
            # Update access time and LRU position
            self._store.move_to_end(series_id)
            entry.last_access = time.monotonic()
            
            # Return sorted copy
            items = list(entry.window)
            items.sort(key=lambda x: x[0])
            return items
    
    def get_values(self, series_id: str) -> Optional[List[T]]:
        """Retorna solo los valores, sin timestamps."""
        items = self.get(series_id)
        if items is None:
            return None
        return [value for _, value in items]
    
    def size(self, series_id: str) -> int:
        """Número de puntos en la ventana actual."""
        with self._lock:
            entry = self._store.get(series_id)
            return len(entry.window) if entry is not None else 0
    
    def clear(self, series_id: str) -> None:
        """Elimina la ventana de series_id."""
        with self._lock:
            if series_id in self._store:
                del self._store[series_id]
    
    def series_ids(self) -> List[str]:
        """Lista de series_id activas."""
        with self._lock:
            return list(self._store.keys())
    
    def get_metrics(self) -> dict:
        """Métricas de uso del cache."""
        with self._lock:
            return {
                "active_series": len(self._store),
                "max_series": self._max_series,
                "window_size": self._window_size,
                "evictions_lru": self._evictions_lru,
                "evictions_ttl": self._evictions_ttl,
                "ttl_seconds": self._ttl,
            }
    
    def _evict_lru_if_full(self) -> None:
        """Elimina la entrada menos recientemente usada si está lleno."""
        while len(self._store) >= self._max_series:
            evicted_id, _ = self._store.popitem(last=False)
            self._evictions_lru += 1
            logger.debug(f"lru_evict: series_id={evicted_id}")
    
    def _evict_expired(self, now: float) -> None:
        """Elimina entradas con TTL vencido."""
        if self._ttl is None:
            return
        
        cutoff = now - self._ttl
        expired = [
            sid for sid, entry in self._store.items()
            if entry.last_access < cutoff
        ]
        
        for sid in expired:
            del self._store[sid]
            self._evictions_ttl += 1
            logger.debug(f"ttl_evict: series_id={sid}")
    
    def append_window(self, window) -> None:
        """Agrega todos los puntos de un TimeWindow a la ventana."""
        series_id = str(window.series_id)
        for point in window.points:
            self.append(series_id, point.value, point.timestamp)
