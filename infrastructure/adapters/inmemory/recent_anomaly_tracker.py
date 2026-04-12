"""InMemoryRecentAnomalyTracker — implementación en memoria del Port.

Fallback cuando Redis no está disponible.
Usa collections.deque con maxlen por serie.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from threading import RLock
from typing import Dict, List, Optional, Tuple

from iot_machine_learning.domain.ports.recent_anomaly_tracker_port import (
    RecentAnomalyTrackerPort,
)

logger = logging.getLogger(__name__)

# Dataclass interna para entradas
class _AnomalyEntry:
    """Entrada de anomalía en memoria."""
    
    def __init__(
        self,
        timestamp: float,
        anomaly_score: float,
        is_anomaly: bool,
        regime: str = "",
    ):
        self.timestamp = timestamp
        self.anomaly_score = anomaly_score
        self.is_anomaly = is_anomaly
        self.regime = regime


class InMemoryRecentAnomalyTracker(RecentAnomalyTrackerPort):
    """Implementación en memoria con LRU eviction.
    
    Estructura: {series_id: deque[_AnomalyEntry]}
    maxlen por serie: 500 entradas (aprox 2 horas a 15s intervalo)
    """
    
    def __init__(self, max_entries_per_series: int | None = None, ttl_seconds: float | None = None) -> None:
        """Inicializar tracker en memoria.

        Args:
            max_entries_per_series: Máximo entradas por serie (None = read from flags)
            ttl_seconds: TTL de entradas antiguas (None = read from flags)
        """
        # Leer desde flags si no se proporcionan explícitamente (hot-reload capability)
        if max_entries_per_series is None or ttl_seconds is None:
            try:
                from ...ml_service.config.feature_flags import get_feature_flags
                flags = get_feature_flags()
                if max_entries_per_series is None:
                    max_entries_per_series = int(flags.ML_ANOMALY_MAX_ENTRIES_PER_SERIES)
                if ttl_seconds is None:
                    ttl_seconds = float(flags.ML_ANOMALY_TTL_SECONDS)
            except Exception:
                # Fallback a valores default
                if max_entries_per_series is None:
                    max_entries_per_series = 500
                if ttl_seconds is None:
                    ttl_seconds = 7200.0

        self._max_entries = max_entries_per_series
        self._ttl_seconds = ttl_seconds

        # {series_id: deque[_AnomalyEntry]}
        self._entries: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_entries_per_series)
        )

        # Contador de consecutivas por serie (se resetea con normales)
        self._consecutive: Dict[str, int] = defaultdict(int)

        self._lock = RLock()

        logger.info(
            "inmemory_tracker_initialized",
            extra={"max_entries": max_entries_per_series, "ttl": ttl_seconds},
        )
    
    def _now(self) -> float:
        """Timestamp actual."""
        return time.time()
    
    def _clean_old_entries(self, series_id: str, cutoff: float) -> None:
        """Eliminar entradas más viejas que cutoff (caller debe tener lock)."""
        entries = self._entries[series_id]
        # Deque no soporta filtrado eficiente, recrear
        new_deque = deque(maxlen=self._max_entries)
        for e in entries:
            if e.timestamp >= cutoff:
                new_deque.append(e)
        self._entries[series_id] = new_deque
    
    def record_anomaly(
        self,
        series_id: str,
        anomaly_score: float,
        timestamp: Optional[float] = None,
        regime: str = "",
    ) -> None:
        """Registrar anomalía."""
        ts = timestamp if timestamp is not None else self._now()
        
        with self._lock:
            # Limpiar entradas antiguas antes de agregar
            cutoff = ts - self._ttl_seconds
            self._clean_old_entries(series_id, cutoff)
            
            # Agregar entrada
            entry = _AnomalyEntry(ts, anomaly_score, True, regime)
            self._entries[series_id].append(entry)
            
            # Incrementar consecutivas
            self._consecutive[series_id] += 1
            
            logger.debug(
                "anomaly_recorded",
                extra={
                    "series_id": series_id,
                    "score": anomaly_score,
                    "consecutive": self._consecutive[series_id],
                },
            )
    
    def record_normal(
        self,
        series_id: str,
        timestamp: Optional[float] = None,
    ) -> None:
        """Registrar predicción normal — resetea consecutivas."""
        ts = timestamp if timestamp is not None else self._now()
        
        with self._lock:
            # Limpiar entradas antiguas
            cutoff = ts - self._ttl_seconds
            self._clean_old_entries(series_id, cutoff)
            
            # Agregar entrada normal
            entry = _AnomalyEntry(ts, 0.0, False, "")
            self._entries[series_id].append(entry)
            
            # Resetear consecutivas
            old_consecutive = self._consecutive[series_id]
            self._consecutive[series_id] = 0
            
            if old_consecutive > 0:
                logger.debug(
                    "consecutive_reset",
                    extra={"series_id": series_id, "previous": old_consecutive},
                )
    
    def get_count_last_n_minutes(self, series_id: str, minutes: int) -> int:
        """Contar anomalías en ventana."""
        with self._lock:
            if series_id not in self._entries:
                return 0
            
            now = self._now()
            cutoff = now - (minutes * 60)
            
            count = 0
            for e in self._entries[series_id]:
                if e.timestamp >= cutoff and e.is_anomaly:
                    count += 1
            
            return count
    
    def get_consecutive_count(self, series_id: str) -> int:
        """Devolver contador de consecutivas."""
        with self._lock:
            return self._consecutive.get(series_id, 0)
    
    def get_anomaly_rate(self, series_id: str, window_minutes: int) -> float:
        """Calcular ratio anomalías / totales en ventana."""
        with self._lock:
            if series_id not in self._entries:
                return 0.0
            
            now = self._now()
            cutoff = now - (window_minutes * 60)
            
            total = 0
            anomalies = 0
            
            for e in self._entries[series_id]:
                if e.timestamp >= cutoff:
                    total += 1
                    if e.is_anomaly:
                        anomalies += 1
            
            if total == 0:
                return 0.0
            
            return anomalies / total
    
    def reset(self, series_id: Optional[str] = None) -> None:
        """Resetear historial."""
        with self._lock:
            if series_id is None:
                self._entries.clear()
                self._consecutive.clear()
                logger.info("all_series_reset")
            else:
                self._entries.pop(series_id, None)
                self._consecutive.pop(series_id, None)
                logger.info("series_reset", extra={"series_id": series_id})
