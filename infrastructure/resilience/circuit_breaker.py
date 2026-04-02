"""Circuit breaker pattern for resilient external service calls.

Implements fail-fast protection for external services like Weaviate and SQL Server.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum, auto
from functools import wraps
from typing import Callable, Optional, TypeVar, Any

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation, calls pass through
    OPEN = auto()        # Failure threshold reached, calls fail fast
    HALF_OPEN = auto()   # Testing if service has recovered


class CircuitBreaker:
    """Circuit breaker for external service protection.
    
    Features:
    - Configurable failure threshold and recovery timeout
    - Thread-safe state transitions
    - Exponential backoff for half-open state
    - Fail-fast when circuit is open
    """
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3,
        success_threshold: int = 2,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.RLock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state (thread-safe)."""
        with self._lock:
            return self._state
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open state."""
        if self._last_failure_time is None:
            return True
        return (time.time() - self._last_failure_time) >= self.recovery_timeout
    
    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to new state with logging."""
        old_state = self._state
        self._state = new_state
        
        if old_state != new_state:
            logger.warning(
                "[CIRCUIT_BREAKER] %s: %s -> %s",
                self.name, old_state.name, new_state.name
            )
            
            # Reset counters on state transitions
            if new_state == CircuitState.CLOSED:
                self._failure_count = 0
                self._success_count = 0
                self._half_open_calls = 0
            elif new_state == CircuitState.OPEN:
                self._half_open_calls = 0
                self._success_count = 0
            elif new_state == CircuitState.HALF_OPEN:
                self._half_open_calls = 0
                self._success_count = 0
    
    def call(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection.
        
        Args:
            fn: Function to call
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Result of fn(*args, **kwargs)
            
        Raises:
            CircuitBreakerOpen: If circuit is open and call is rejected
            Exception: Any exception from fn
        """
        with self._lock:
            # Check if we should transition from OPEN to HALF_OPEN
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
                else:
                    raise CircuitBreakerOpen(
                        f"Circuit {self.name} is OPEN. Call rejected."
                    )
            
            # In HALF_OPEN, limit concurrent test calls
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls >= self.half_open_max_calls:
                    raise CircuitBreakerOpen(
                        f"Circuit {self.name} is HALF_OPEN with max calls reached."
                    )
                self._half_open_calls += 1
        
        # Execute the call (outside lock to allow concurrent calls in CLOSED state)
        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self) -> None:
        """Record successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            else:
                # In CLOSED state, reset failure count on success
                self._failure_count = max(0, self._failure_count - 1)
    
    def _on_failure(self) -> None:
        """Record failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.HALF_OPEN:
                # Failure in half-open immediately opens circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)
    
    def get_metrics(self) -> dict:
        """Get circuit breaker metrics."""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.name,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "half_open_calls": self._half_open_calls,
                "last_failure_time": self._last_failure_time,
            }


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Global circuit breaker registry
_circuit_breakers: dict[str, CircuitBreaker] = {}
_registry_lock = threading.Lock()


def get_circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
) -> CircuitBreaker:
    """Get or create a circuit breaker by name."""
    with _registry_lock:
        if name not in _circuit_breakers:
            _circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
            )
        return _circuit_breakers[name]


def circuit_breaker(
    name: str,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
) -> Callable:
    """Decorator to add circuit breaker to a function.
    
    Usage:
        @circuit_breaker("weaviate", failure_threshold=3)
        def call_weaviate(query: str) -> dict:
            ...
    """
    cb = get_circuit_breaker(name, failure_threshold, recovery_timeout)
    
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @wraps(fn)
        def wrapper(*args, **kwargs) -> T:
            return cb.call(fn, *args, **kwargs)
        return wrapper
    return decorator
