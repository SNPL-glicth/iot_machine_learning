"""Backpressure controller for ML stream consumer.

Priority-based admission control: rejects messages when system overloaded.
"""
from __future__ import annotations

import threading

# Backpressure defaults
DEFAULT_MAX_IN_FLIGHT = 1000
DEFAULT_LATENCY_TARGET_MS = 5000


class BackpressureController:
    """Simple backpressure: rejects messages when system overloaded."""

    def __init__(
        self,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        target_latency_ms: float = DEFAULT_LATENCY_TARGET_MS,
    ):
        self._max_in_flight = max_in_flight
        self._target_latency = target_latency_ms
        self._in_flight = 0
        self._lock = threading.Lock()
        self._rejected = 0
        self._accepted = 0

    def can_accept(self, priority: str = "normal") -> bool:
        """Check if we can accept a new message.

        Priority levels:
        - critical: always accepted (unless >90% capacity)
        - high: accepted until 80% capacity
        - normal: accepted until 70% capacity
        - low: accepted until 50% capacity
        """
        with self._lock:
            load = self._in_flight / self._max_in_flight

            if priority == "critical":
                return load < 0.9
            elif priority == "high":
                return load < 0.8
            elif priority == "normal":
                return load < 0.7
            else:  # low
                return load < 0.5

    def record_start(self):
        """Call when starting to process a message."""
        with self._lock:
            self._in_flight += 1
            self._accepted += 1

    def record_complete(self, latency_ms: float):
        """Call when finished processing."""
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def record_reject(self):
        """Call when rejecting a message."""
        with self._lock:
            self._rejected += 1

    def get_metrics(self) -> dict:
        """Get current backpressure metrics."""
        with self._lock:
            return {
                "in_flight": self._in_flight,
                "max_in_flight": self._max_in_flight,
                "load_factor": self._in_flight / self._max_in_flight,
                "accepted": self._accepted,
                "rejected": self._rejected,
            }
