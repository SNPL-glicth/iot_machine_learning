"""Async prediction worker pool — FIX P3-1: decouples ingestion from prediction.

N threads consume from an internal queue and execute predictions in parallel.
Environment variables:
  ML_PREDICTION_WORKERS — number of worker threads (default: 4)
  ML_PREDICTION_QUEUE_MAX — max queue size (default: 5000)
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_N_WORKERS = int(os.environ.get("ML_PREDICTION_WORKERS", "4"))
DEFAULT_QUEUE_MAX = int(os.environ.get("ML_PREDICTION_QUEUE_MAX", "5000"))


@dataclass
class PredictionTask:
    sensor_id: int
    prediction_id: Optional[str] = None
    mode: str = "stream"  # "stream" | "http"
    horizon_minutes: int = 10
    window: int = 60
    dedupe_minutes: int = 10
    extra: dict = field(default_factory=dict)


class PredictionWorker:
    """Worker pool that decouples ingestion from prediction execution."""

    def __init__(
        self,
        n_workers: int = DEFAULT_N_WORKERS,
        queue_max: int = DEFAULT_QUEUE_MAX,
        predict_fn: Optional[Callable[[PredictionTask], Any]] = None,
        worker_factory: Optional[Callable[[], Callable[[PredictionTask], Any]]] = None,
        result_store: Optional[Any] = None,
    ) -> None:
        self._n_workers = n_workers
        self._queue: queue.Queue[PredictionTask] = queue.Queue(maxsize=queue_max)
        self._predict_fn = predict_fn
        self._worker_factory = worker_factory
        self._result_store = result_store
        self._workers: list[threading.Thread] = []
        self._running = False
        self._processed = 0
        self._errors = 0
        self._dropped = 0
        self._lock = threading.Lock()
        logger.info(
            "[PREDICTION_WORKER] Initialized workers=%d queue_max=%d",
            n_workers, queue_max,
        )

    def enqueue(self, task: PredictionTask) -> bool:
        """Enqueue a task. Returns False if queue is full (backpressure)."""
        try:
            self._queue.put_nowait(task)
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
            logger.warning(
                "[PREDICTION_WORKER] Queue full, dropped sensor=%s",
                task.sensor_id,
            )
            return False

    def is_healthy(self) -> bool:
        """False if queue >90% capacity (all workers likely blocked)."""
        return self._queue.qsize() <= self._queue.maxsize * 0.9

    def start(self) -> None:
        self._running = True
        for i in range(self._n_workers):
            t = threading.Thread(
                target=self._worker_loop, args=(i,),
                daemon=True, name=f"PredictionWorker-{i}",
            )
            t.start()
            self._workers.append(t)
        logger.info("[PREDICTION_WORKER] Started %d workers", self._n_workers)

    def stop(self, timeout: float = 30.0) -> None:
        self._running = False
        deadline = time.monotonic() + timeout
        for t in self._workers:
            t.join(timeout=max(0, deadline - time.monotonic()))
        alive = sum(1 for t in self._workers if t.is_alive())
        if alive:
            logger.warning("[PREDICTION_WORKER] %d workers did not stop", alive)
        logger.info(
            "[PREDICTION_WORKER] Stopped processed=%d errors=%d dropped=%d",
            self._processed, self._errors, self._dropped,
        )

    def _worker_loop(self, worker_id: int) -> None:
        predict_fn = self._predict_fn
        if self._worker_factory is not None:
            predict_fn = self._worker_factory()
        while self._running:
            try:
                task = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                result = None
                if predict_fn is not None:
                    result = predict_fn(task)
                with self._lock:
                    self._processed += 1
                if self._result_store is not None and task.prediction_id is not None:
                    self._result_store.set(task.prediction_id, result)
            except Exception as e:
                with self._lock:
                    self._errors += 1
                logger.error(
                    "[PREDICTION_WORKER] worker=%d sensor=%s error: %s",
                    worker_id, task.sensor_id, e,
                )
                if self._result_store is not None and task.prediction_id is not None:
                    self._result_store.set(task.prediction_id, {"error": str(e)})
            finally:
                self._queue.task_done()
