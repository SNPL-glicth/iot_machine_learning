"""Circuit breaker implementation for Redis operations.

Extracted from circuit.py as part of modularization (<180 lines per file).
Core CircuitBreaker class without factory functions.
"""

from __future__ import annotations

import logging
import random
import time
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

try:
    from prometheus_client import Counter, Gauge
except ImportError:  # pragma: no cover
    Counter = None  # type: ignore[misc, assignment]
    Gauge = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

if Counter is not None and Gauge is not None:
    CIRCUIT_STATE_TRANSITIONS = Counter(
        "zenin_circuit_breaker_transitions_total",
        "Total de transiciones de estado del circuit breaker",
        ["breaker_name", "from_state", "to_state"],
    )
    CIRCUIT_BREAKER_STATE = Gauge(
        "zenin_circuit_breaker_state",
        "Estado actual del circuit breaker (0=CLOSED, 1=OPEN, 2=HALF_OPEN)",
        ["breaker_name"],
    )
else:
    CIRCUIT_STATE_TRANSITIONS = None  # type: ignore[assignment]
    CIRCUIT_BREAKER_STATE = None  # type: ignore[assignment]

_STATE_VALUE_MAP = {
    "closed": 0,
    "open": 1,
    "half_open": 2,
}

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"       # Normal operation
    OPEN = "open"         # Failing, reject fast
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and no fallback provided."""
    pass


class _CircuitStateTransition:
    """Handles circuit breaker state transitions (RES-CRIT-1 SRP).
    
    Applies SRP: State transition logic is separate concern.
    """
    
    @staticmethod
    def should_reopen_from_half_open(
        half_open_failures: int,
        failure_budget: int,
    ) -> bool:
        """Check if circuit should reopen from half-open.
        
        Args:
            half_open_failures: Current failure count in half-open.
            failure_budget: Maximum failures allowed before reopening.
        
        Returns:
            True if circuit should reopen.
        
        Applies RES-CRIT-1: Tolerates failures up to budget.
        """
        return half_open_failures >= failure_budget
    
    @staticmethod
    def should_close_from_half_open(
        half_open_successes: int,
        required_successes: int,
    ) -> bool:
        """Check if circuit should close from half-open.
        
        Args:
            half_open_successes: Current success count in half-open.
            required_successes: Successes needed to close.
        
        Returns:
            True if circuit should close.
        """
        return half_open_successes >= required_successes
    
    @staticmethod
    def should_open_from_closed(
        failure_count: int,
        failure_threshold: int,
    ) -> bool:
        """Check if circuit should open from closed.
        
        Args:
            failure_count: Current failure count.
            failure_threshold: Threshold to open.
        
        Returns:
            True if circuit should open.
        """
        return failure_count >= failure_threshold


class CircuitBreaker:
    """Circuit breaker for Redis operations.
    
    Prevents cascading failures by:
    1. Detecting consecutive failures
    2. Opening circuit after threshold (fail fast)
    3. Testing recovery after timeout (half-open)
    4. Closing circuit on success
    
    Args:
        name: Circuit name for logging/metrics
        failure_threshold: Consecutive failures before opening (default: 5)
        recovery_timeout: Seconds before testing recovery (default: 30)
        half_open_max_calls: Successes needed to close (default: 3)
        expected_exception: Exception type that counts as failure
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: int = 30,
        half_open_max_calls: int = 3,
        expected_exception: type = Exception,
        half_open_failure_budget: int = 2,  # RES-CRIT-1
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.expected_exception = expected_exception
        self.half_open_failure_budget = half_open_failure_budget  # RES-CRIT-1
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: Optional[float] = None
        self._half_open_successes = 0
        self._half_open_failures = 0  # RES-CRIT-1

        if CIRCUIT_BREAKER_STATE is not None:
            CIRCUIT_BREAKER_STATE.labels(breaker_name=self.name).set(
                _STATE_VALUE_MAP.get(CircuitState.CLOSED.value, -1)
            )

        logger.debug(
            "circuit_breaker_created",
            extra={
                "circuit_name": name,
                "threshold": failure_threshold,
                "recovery_timeout": recovery_timeout,
            }
        )
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        return self._state
    
    def call(
        self,
        operation: Callable[[], T],
        fallback: Optional[Callable[[], T]] = None,
    ) -> T:
        """Execute operation with circuit breaker protection."""
        # Check if we should transition from OPEN to HALF_OPEN
        if self._state == CircuitState.OPEN:
            if self._should_attempt_reset():
                prev = self._state
                logger.info("circuit_half_open", extra={"circuit_name": self.name})
                self._state = CircuitState.HALF_OPEN
                self._half_open_successes = 0
                self._half_open_failures = 0  # RES-CRIT-1: Reset failure counter
                self._record_transition(prev, CircuitState.HALF_OPEN)
            else:
                logger.debug("circuit_open_reject", extra={"circuit_name": self.name})
                if fallback:
                    return fallback()
                raise CircuitOpenError(
                    f"Circuit breaker '{self.name}' is OPEN. Redis unavailable."
                )
        
        # Execute operation
        try:
            result = operation()
            self._on_success()
            return result
            
        except self.expected_exception as e:
            self._on_failure()
            
            if fallback:
                logger.debug(
                    "circuit_fallback_executed",
                    extra={
                        "circuit_name": self.name,
                        "error": str(e),
                        "state": self._state.value,
                    }
                )
                return fallback()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to test recovery (with jitter)."""
        if self._last_failure_time is None:
            return True
        jitter = random.uniform(0, self.recovery_timeout * 0.1)
        return (time.time() - self._last_failure_time) >= (self.recovery_timeout + jitter)
    
    def _on_success(self) -> None:
        """Handle successful operation."""
        if self._state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            
            if self._half_open_successes >= self.half_open_max_calls:
                prev = self._state
                logger.info("circuit_closed", extra={"circuit_name": self.name})
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._half_open_successes = 0
                self._record_transition(prev, CircuitState.CLOSED)
        else:
            self._failure_count = max(0, self._failure_count - 1)
    
    def _on_failure(self) -> None:
        """Handle failed operation."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.HALF_OPEN:
            # RES-CRIT-1: Tolerate failures up to budget
            self._half_open_failures += 1
            
            if _CircuitStateTransition.should_reopen_from_half_open(
                self._half_open_failures,
                self.half_open_failure_budget,
            ):
                prev = self._state
                logger.warning(
                    "circuit_reopened",
                    extra={
                        "circuit_name": self.name,
                        "failure_count": self._failure_count,
                        "half_open_failures": self._half_open_failures,
                        "failure_budget": self.half_open_failure_budget,
                    }
                )
                self._state = CircuitState.OPEN
                self._half_open_failures = 0  # Reset for next half-open
                self._record_transition(prev, CircuitState.OPEN)
            else:
                logger.debug(
                    "circuit_half_open_failure_tolerated",
                    extra={
                        "circuit_name": self.name,
                        "half_open_failures": self._half_open_failures,
                        "failure_budget": self.half_open_failure_budget,
                    }
                )
            
        elif _CircuitStateTransition.should_open_from_closed(
            self._failure_count,
            self.failure_threshold,
        ):
            prev = self._state
            logger.error(
                "circuit_opened",
                extra={
                    "circuit_name": self.name,
                    "failure_count": self._failure_count,
                    "threshold": self.failure_threshold,
                }
            )
            self._state = CircuitState.OPEN
            self._record_transition(prev, CircuitState.OPEN)
    
    def force_open(self) -> None:
        """Manually open the circuit (for maintenance/testing)."""
        self._state = CircuitState.OPEN
        self._last_failure_time = time.time()
        logger.warning("circuit_force_opened", extra={"circuit_name": self.name})
    
    def force_close(self) -> None:
        """Manually close the circuit (after fixing issue)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_successes = 0
        logger.info("circuit_force_closed", extra={"circuit_name": self.name})
    
    def _record_transition(self, prev: CircuitState, new: CircuitState) -> None:
        """Record state transition in Prometheus metrics."""
        if CIRCUIT_STATE_TRANSITIONS is not None:
            CIRCUIT_STATE_TRANSITIONS.labels(
                breaker_name=self.name,
                from_state=prev.value,
                to_state=new.value,
            ).inc()
        if CIRCUIT_BREAKER_STATE is not None:
            CIRCUIT_BREAKER_STATE.labels(
                breaker_name=self.name,
            ).set(_STATE_VALUE_MAP.get(new.value, -1))

    def get_metrics(self) -> dict:
        """Get circuit breaker metrics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "half_open_successes": self._half_open_successes,
            "last_failure": self._last_failure_time,
            "time_since_last_failure": (
                time.time() - self._last_failure_time
                if self._last_failure_time else None
            ),
        }
