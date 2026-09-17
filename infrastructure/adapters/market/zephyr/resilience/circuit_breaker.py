"""Circuit Breaker implementation for high-frequency and live market broker interactions."""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

_T = TypeVar("_T")


class CircuitState(str, Enum):
    CLOSED = "CLOSED"      # Normal: requests pass through
    OPEN = "OPEN"          # Tripped: requests fail fast without network call
    HALF_OPEN = "HALF_OPEN"# Testing recovery: limited test traffic allowed


class CircuitBreakerOpenError(RuntimeError):
    """Raised when request is rejected by an open circuit breaker."""


class CircuitBreaker:
    """Non-blocking state machine protecting broker REST APIs from cascade failures."""

    def __init__(
        self,
        name: str = "broker_api",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_successes_needed: int = 2,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_successes_needed = half_open_successes_needed

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_state_change = time.time()
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN and (time.time() - self._last_state_change >= self.recovery_timeout):
            self._state = CircuitState.HALF_OPEN
            self._last_state_change = time.time()
            self._success_count = 0
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def can_execute(self) -> bool:
        """Evaluates whether a request may proceed based on current state and timeout."""
        now = time.time()
        if self._state == CircuitState.CLOSED:
            return True
        if self._state == CircuitState.OPEN:
            if now - self._last_state_change >= self.recovery_timeout:
                logger.info("CircuitBreaker [%s] transitioned OPEN -> HALF_OPEN (timeout elapsed)", self.name)
                self._state = CircuitState.HALF_OPEN
                self._last_state_change = now
                self._success_count = 0
                return True
            return False
        # HALF_OPEN allows probationary requests
        return True

    def record_success(self) -> None:
        """Records a successful operation."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_successes_needed:
                logger.info("CircuitBreaker [%s] recovered HALF_OPEN -> CLOSED", self.name)
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
                self._last_state_change = time.time()
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0

    def record_failure(self, exc: Exception | None = None) -> None:
        """Records a failed operation and trips the breaker if threshold is exceeded."""
        self._failure_count += 1
        now = time.time()
        if self._state == CircuitState.HALF_OPEN:
            logger.warning("CircuitBreaker [%s] failed probe in HALF_OPEN. Tripping back to OPEN. Error: %s", self.name, exc)
            self._state = CircuitState.OPEN
            self._last_state_change = now
        elif self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
            logger.error("CircuitBreaker [%s] failure threshold %d reached. Tripping to OPEN! Error: %s", self.name, self._failure_count, exc)
            self._state = CircuitState.OPEN
            self._last_state_change = now

    async def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Executes an async callable wrapped in the circuit breaker."""
        if not self.can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. Broker requests blocked until cooldown expires."
            )
        try:
            res = await fn(*args, **kwargs)
            self.record_success()
            return res
        except Exception as exc:
            self.record_failure(exc)
            raise

    def get_metrics(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "last_state_change": self._last_state_change,
        }
