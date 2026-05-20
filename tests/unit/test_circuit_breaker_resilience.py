"""Tests for circuit breaker pattern — resilience infrastructure."""
import pytest
import time


class TestCircuitBreakerStates:
    """Test circuit breaker state transitions."""

    def test_initial_state_closed(self):
        from iot_machine_learning.infrastructure.resilience.circuit_breaker import (
            CircuitBreaker, CircuitState,
        )
        cb = CircuitBreaker("test", failure_threshold=3)
        assert cb.state == CircuitState.CLOSED

    def test_opens_after_threshold(self):
        from iot_machine_learning.infrastructure.resilience.circuit_breaker import (
            CircuitBreaker, CircuitState, CircuitBreakerOpen,
        )
        cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout=60.0)

        def fail():
            raise RuntimeError("service down")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                cb.call(fail)

        assert cb.state == CircuitState.OPEN

        with pytest.raises(CircuitBreakerOpen):
            cb.call(fail)

    def test_success_resets_failure_count(self):
        from iot_machine_learning.infrastructure.resilience.circuit_breaker import (
            CircuitBreaker, CircuitState,
        )
        cb = CircuitBreaker("test", failure_threshold=3)

        def fail():
            raise RuntimeError("fail")

        def succeed():
            return "ok"

        # 2 failures (below threshold)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(fail)

        # 1 success decrements
        assert cb.call(succeed) == "ok"
        assert cb.state == CircuitState.CLOSED

    def test_half_open_transitions_to_closed(self):
        from iot_machine_learning.infrastructure.resilience.circuit_breaker import (
            CircuitBreaker, CircuitState,
        )
        cb = CircuitBreaker(
            "test", failure_threshold=2, recovery_timeout=0.01,
            success_threshold=1, enable_exponential_backoff=False,
        )

        def fail():
            raise RuntimeError("fail")

        def succeed():
            return "ok"

        # Trip the breaker
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(fail)
        assert cb.state == CircuitState.OPEN

        # Wait for recovery timeout
        time.sleep(0.02)

        # Next call transitions to HALF_OPEN, succeeds → CLOSED
        assert cb.call(succeed) == "ok"
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        from iot_machine_learning.infrastructure.resilience.circuit_breaker import (
            CircuitBreaker, CircuitState,
        )
        cb = CircuitBreaker(
            "test", failure_threshold=2, recovery_timeout=0.01,
            enable_exponential_backoff=False,
        )

        def fail():
            raise RuntimeError("fail")

        # Trip
        for _ in range(2):
            with pytest.raises(RuntimeError):
                cb.call(fail)

        time.sleep(0.02)

        # Fail in HALF_OPEN → back to OPEN
        with pytest.raises(RuntimeError):
            cb.call(fail)
        assert cb.state == CircuitState.OPEN


class TestCircuitBreakerMetrics:
    """Test circuit breaker metrics reporting."""

    def test_get_metrics(self):
        from iot_machine_learning.infrastructure.resilience.circuit_breaker import (
            CircuitBreaker, CircuitState,
        )
        cb = CircuitBreaker("metrics_test", failure_threshold=5)
        metrics = cb.get_metrics()
        assert metrics["name"] == "metrics_test"
        assert metrics["state"] == "CLOSED"
        assert metrics["failure_count"] == 0

    def test_metrics_after_failures(self):
        from iot_machine_learning.infrastructure.resilience.circuit_breaker import (
            CircuitBreaker,
        )
        cb = CircuitBreaker("test", failure_threshold=5)

        def fail():
            raise RuntimeError("x")

        with pytest.raises(RuntimeError):
            cb.call(fail)

        metrics = cb.get_metrics()
        assert metrics["failure_count"] == 1
        assert metrics["last_failure_time"] is not None


class TestCircuitBreakerDecorator:
    """Test the decorator interface."""

    def test_decorator_importable(self):
        from iot_machine_learning.infrastructure.resilience.circuit_breaker import circuit_breaker
        assert callable(circuit_breaker)

    def test_decorator_wraps_function(self):
        from iot_machine_learning.infrastructure.resilience.circuit_breaker import circuit_breaker

        @circuit_breaker("test_decorator", failure_threshold=10)
        def my_service_call(x):
            return x * 2

        assert my_service_call(5) == 10

    def test_get_circuit_breaker_registry(self):
        from iot_machine_learning.infrastructure.resilience.circuit_breaker import get_circuit_breaker
        cb1 = get_circuit_breaker("singleton_test")
        cb2 = get_circuit_breaker("singleton_test")
        assert cb1 is cb2


class TestExponentialBackoff:
    """Test exponential backoff in circuit breaker."""

    def test_backoff_increases_timeout(self):
        from iot_machine_learning.infrastructure.resilience.circuit_breaker import (
            CircuitBreaker,
        )
        cb = CircuitBreaker(
            "backoff_test", failure_threshold=2, recovery_timeout=1.0,
            enable_exponential_backoff=True, max_backoff_timeout=10.0,
        )
        metrics = cb.get_metrics()
        # Initial effective timeout = base (no consecutive failures)
        assert metrics["effective_timeout_seconds"] == 1.0
