"""Background job that periodically computes Pearson correlations.

Runs in a daemon thread.  Populates SqlCorrelationAdapter's cache so the
PERCEIVE phase never blocks on SQL.  Failures are logged and swallowed;
the prediction pipeline is never affected.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

_DEFAULT_INTERVAL_SECONDS = 3600


class CorrelationBackgroundJob:
    """Periodically refreshes the correlation cache for all active series.

    Usage::

        job = CorrelationBackgroundJob(adapter=sql_correlation_adapter,
                                       storage=storage_port)
        job.start()          # non-blocking, daemon thread
        ...
        job.stop()           # graceful shutdown
    """

    def __init__(
        self,
        adapter,
        storage,
        interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
        max_refresh_per_cycle: int = 100,
    ) -> None:
        self._adapter = adapter
        self._storage = storage
        self._interval = interval_seconds
        self._max_refresh = max_refresh_per_cycle
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="correlation-bg-job",
            daemon=True,
        )
        self._thread.start()
        logger.info("correlation_background_job_started interval_s=%d", self._interval)

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("correlation_background_job_stopped")

    def run_once(self) -> None:
        """Compute correlations for all active series. Public for testing."""
        try:
            series_ids = self._storage.list_active_series_ids()
        except Exception as exc:
            logger.error("correlation_job_list_failed: %s", exc)
            return

        priorities = []
        for sid in series_ids:
            try:
                cache_age = self._adapter.get_cache_age(sid)
            except Exception:
                cache_age = float('inf')
            
            volatility = 0.0
            gradient = 0.0
            try:
                sensor_id = int(sid) if sid.isdigit() else 0
                if sensor_id > 0:
                    window = self._storage.load_sensor_window(sensor_id, limit=10)
                    if len(window.readings) >= 3:
                        values = [r.value for r in window.readings]
                        import math
                        mean_val = sum(values) / len(values)
                        volatility = math.sqrt(sum((v - mean_val) ** 2 for v in values) / len(values))
                        gradient = abs(values[-1] - values[0]) / max(len(values) - 1, 1)
            except Exception:
                pass
            
            priority = cache_age * (1.0 + volatility + gradient)
            priorities.append((priority, sid))
        
        priorities.sort(reverse=True, key=lambda x: x[0])
        
        top_series = [sid for _, sid in priorities[:self._max_refresh]]
        total_series = len(series_ids)

        refreshed, failed = 0, 0
        for sid in top_series:
            if self._stop_event.is_set():
                break
            try:
                self._adapter.refresh_series(sid)
                refreshed += 1
            except Exception as exc:
                failed += 1
                logger.debug("correlation_refresh_failed series=%s: %s", sid, exc)

        logger.info(
            "correlation_job_done refreshed=%d failed=%d total=%d", refreshed, failed, total_series
        )

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self.run_once()
            self._stop_event.wait(timeout=self._interval)
