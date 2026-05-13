"""Tests for RES-CRIT-1, COG-CRIT-1, COG-CRIT-2, COG-SEV-4 fixes.

Validates resilience and cognitive layer fixes.
"""

from __future__ import annotations

import time
from unittest.mock import Mock

import pytest

from iot_machine_learning.infrastructure.persistence.redis.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitOpenError,
    _CircuitStateTransition,
)


class TestResCrit1CircuitBreakerTolerance:
    """RES-CRIT-1: Circuit breaker must tolerate failures in half-open."""
    
    def test_single_failure_in_half_open_reopens_immediately_before_fix(self):
        """BEFORE FIX: Single failure in half-open reopens circuit immediately."""
        # Create circuit with budget=1 (old behavior)
        cb = CircuitBreaker(
            name="test_redis",
            failure_threshold=3,
            recovery_timeout=1,
            half_open_failure_budget=1,  # Single failure reopens
        )
        
        # Open the circuit
        for _ in range(3):
            try:
                cb.call(lambda: self._failing_operation())
            except Exception:
                pass
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # First call transitions to HALF_OPEN
        try:
            cb.call(lambda: self._failing_operation())
        except Exception:
            pass
        
        # With budget=1, circuit should reopen immediately
        assert cb.state == CircuitState.OPEN
    
    def test_tolerates_failures_up_to_budget_after_fix(self):
        """AFTER FIX: Circuit tolerates failures up to budget in half-open."""
        # Create circuit with budget=2 (new behavior)
        cb = CircuitBreaker(
            name="test_redis",
            failure_threshold=3,
            recovery_timeout=1,
            half_open_failure_budget=2,  # Tolerates 2 failures
        )
        
        # Open the circuit
        for _ in range(3):
            try:
                cb.call(lambda: self._failing_operation())
            except Exception:
                pass
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery timeout
        time.sleep(1.1)
        
        # First call transitions to HALF_OPEN
        try:
            cb.call(lambda: self._failing_operation())
        except Exception:
            pass
        
        # Should still be HALF_OPEN (failure 1/2)
        assert cb.state == CircuitState.HALF_OPEN
        
        # Second failure should still keep it HALF_OPEN (failure 2/2)
        try:
            cb.call(lambda: self._failing_operation())
        except Exception:
            pass
        
        # Now at budget limit, should reopen
        assert cb.state == CircuitState.OPEN
    
    def test_state_transition_logic_isolated(self):
        """State transition logic is isolated in _CircuitStateTransition (SRP)."""
        # Test should_reopen_from_half_open
        assert _CircuitStateTransition.should_reopen_from_half_open(0, 2) is False
        assert _CircuitStateTransition.should_reopen_from_half_open(1, 2) is False
        assert _CircuitStateTransition.should_reopen_from_half_open(2, 2) is True
        assert _CircuitStateTransition.should_reopen_from_half_open(3, 2) is True
        
        # Test should_close_from_half_open
        assert _CircuitStateTransition.should_close_from_half_open(2, 3) is False
        assert _CircuitStateTransition.should_close_from_half_open(3, 3) is True
        
        # Test should_open_from_closed
        assert _CircuitStateTransition.should_open_from_closed(4, 5) is False
        assert _CircuitStateTransition.should_open_from_closed(5, 5) is True
    
    def test_successful_recovery_after_partial_failures(self):
        """Circuit can recover even after some failures in half-open."""
        cb = CircuitBreaker(
            name="test_redis",
            failure_threshold=3,
            recovery_timeout=1,
            half_open_max_calls=2,
            half_open_failure_budget=2,
        )
        
        # Open the circuit
        for _ in range(3):
            try:
                cb.call(lambda: self._failing_operation())
            except Exception:
                pass
        
        assert cb.state == CircuitState.OPEN
        
        # Wait for recovery
        time.sleep(1.1)
        
        # Transition to HALF_OPEN with one failure (tolerated)
        try:
            cb.call(lambda: self._failing_operation())
        except Exception:
            pass
        
        assert cb.state == CircuitState.HALF_OPEN
        
        # Now succeed twice to close
        cb.call(lambda: "success")
        assert cb.state == CircuitState.HALF_OPEN
        
        cb.call(lambda: "success")
        assert cb.state == CircuitState.CLOSED  # Recovered!
    
    def test_failure_counter_resets_on_half_open_entry(self):
        """Failure counter should reset when entering half-open."""
        cb = CircuitBreaker(
            name="test_redis",
            failure_threshold=2,
            recovery_timeout=1,
            half_open_failure_budget=2,
        )
        
        # Open circuit
        for _ in range(2):
            try:
                cb.call(lambda: self._failing_operation())
            except Exception:
                pass
        
        assert cb.state == CircuitState.OPEN
        
        # Wait and enter half-open
        time.sleep(1.1)
        try:
            cb.call(lambda: self._failing_operation())
        except Exception:
            pass
        
        # Should be in half-open with failure counter = 1
        assert cb.state == CircuitState.HALF_OPEN
        
        # One more failure should still be tolerated
        try:
            cb.call(lambda: self._failing_operation())
        except Exception:
            pass
        
        # Should reopen now (reached budget)
        assert cb.state == CircuitState.OPEN
    
    def test_configurable_failure_budget(self):
        """Failure budget should be configurable."""
        # Budget = 1 (strict)
        cb_strict = CircuitBreaker(
            name="strict",
            failure_threshold=2,
            recovery_timeout=1,
            half_open_failure_budget=1,
        )
        
        # Budget = 5 (lenient)
        cb_lenient = CircuitBreaker(
            name="lenient",
            failure_threshold=2,
            recovery_timeout=1,
            half_open_failure_budget=5,
        )
        
        # Open both
        for cb in [cb_strict, cb_lenient]:
            for _ in range(2):
                try:
                    cb.call(lambda: self._failing_operation())
                except Exception:
                    pass
        
        time.sleep(1.1)
        
        # Strict: 1 failure reopens
        try:
            cb_strict.call(lambda: self._failing_operation())
        except Exception:
            pass
        assert cb_strict.state == CircuitState.OPEN
        
        # Lenient: 1 failure keeps it half-open
        try:
            cb_lenient.call(lambda: self._failing_operation())
        except Exception:
            pass
        assert cb_lenient.state == CircuitState.HALF_OPEN
    
    @staticmethod
    def _failing_operation():
        """Helper that always fails."""
        raise Exception("Redis connection failed")
