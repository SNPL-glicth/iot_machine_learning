"""Tests for EnsembleWatchdog - Fase 11."""

import pytest

from core.ensemble.ensemble_watchdog import (
    EnsembleHealth,
    EnsembleWatchdog,
    WatchdogSnapshot,
)


class TestEnsembleWatchdogInit:
    """Tests for EnsembleWatchdog initialization."""

    def test_init_default_params(self):
        """Watchdog initializes with default parameters."""
        watchdog = EnsembleWatchdog()
        assert watchdog._min_active_ratio == 0.3
        assert watchdog._min_weight_threshold == 0.05
        assert watchdog._suppression_threshold == 0.8
        assert watchdog._max_suppressed_ratio == 0.5


class TestEnsembleWatchdogEvaluation:
    """Tests for ensemble health evaluation."""

    def test_healthy_ensemble(self):
        """Todos engines activos con pesos altos → HEALTHY."""
        watchdog = EnsembleWatchdog()
        weights = {"taylor": 0.4, "statistical": 0.3, "baseline": 0.3}
        suppressions = {"taylor": 0.1, "statistical": 0.1, "baseline": 0.1}
        
        snapshot = watchdog.evaluate(weights, suppressions)
        
        assert snapshot.health == EnsembleHealth.HEALTHY
        assert snapshot.active_engines == 3
        assert snapshot.total_engines == 3
        assert snapshot.recovery_recommended is False

    def test_degraded_ensemble(self):
        """40% engines activos → DEGRADED."""
        watchdog = EnsembleWatchdog()
        weights = {"taylor": 0.4, "statistical": 0.04, "baseline": 0.04}
        suppressions = {"taylor": 0.1, "statistical": 0.9, "baseline": 0.9}
        
        snapshot = watchdog.evaluate(weights, suppressions)
        
        assert snapshot.health == EnsembleHealth.DEGRADED
        assert snapshot.active_engines == 1
        assert snapshot.recovery_recommended is False

    def test_critical_ensemble(self):
        """20% engines activos → CRITICAL."""
        watchdog = EnsembleWatchdog()
        weights = {"taylor": 0.04, "statistical": 0.04, "baseline": 0.04, "seasonal": 0.88}
        suppressions = {"taylor": 0.9, "statistical": 0.9, "baseline": 0.9, "seasonal": 0.1}
        
        snapshot = watchdog.evaluate(weights, suppressions)
        
        assert snapshot.health == EnsembleHealth.CRITICAL
        assert snapshot.active_engines == 1
        assert snapshot.recovery_recommended is True

    def test_collapsed_ensemble(self):
        """0 engines activos o todos en min_weight → COLLAPSED."""
        watchdog = EnsembleWatchdog()
        weights = {"taylor": 0.01, "statistical": 0.01, "baseline": 0.01}
        suppressions = {"taylor": 0.95, "statistical": 0.95, "baseline": 0.95}
        
        snapshot = watchdog.evaluate(weights, suppressions)
        
        assert snapshot.health == EnsembleHealth.COLLAPSED
        assert snapshot.active_engines == 0
        assert snapshot.recovery_recommended is True

    def test_should_trigger_recovery_critical(self):
        """CRITICAL health should trigger recovery."""
        watchdog = EnsembleWatchdog()
        weights = {"taylor": 0.04, "statistical": 0.04, "baseline": 0.04}
        suppressions = {"taylor": 0.9, "statistical": 0.9, "baseline": 0.9}
        
        snapshot = watchdog.evaluate(weights, suppressions)
        
        assert watchdog.should_trigger_recovery(snapshot) is True

    def test_should_trigger_recovery_collapsed(self):
        """COLLAPSED health should trigger recovery."""
        watchdog = EnsembleWatchdog()
        weights = {"taylor": 0.01, "statistical": 0.01, "baseline": 0.01}
        suppressions = {"taylor": 0.95, "statistical": 0.95, "baseline": 0.95}
        
        snapshot = watchdog.evaluate(weights, suppressions)
        
        assert watchdog.should_trigger_recovery(snapshot) is True

    def test_no_trigger_healthy(self):
        """HEALTHY should not trigger recovery."""
        watchdog = EnsembleWatchdog()
        weights = {"taylor": 0.4, "statistical": 0.3, "baseline": 0.3}
        suppressions = {"taylor": 0.1, "statistical": 0.1, "baseline": 0.1}
        
        snapshot = watchdog.evaluate(weights, suppressions)
        
        assert watchdog.should_trigger_recovery(snapshot) is False

    def test_no_trigger_degraded(self):
        """DEGRADED should not trigger recovery."""
        watchdog = EnsembleWatchdog()
        weights = {"taylor": 0.4, "statistical": 0.04, "baseline": 0.04}
        suppressions = {"taylor": 0.1, "statistical": 0.9, "baseline": 0.9}
        
        snapshot = watchdog.evaluate(weights, suppressions)
        
        assert watchdog.should_trigger_recovery(snapshot) is False

    def test_snapshot_fields_populated(self):
        """All snapshot fields are populated correctly."""
        watchdog = EnsembleWatchdog()
        weights = {"taylor": 0.4, "statistical": 0.3, "baseline": 0.3}
        suppressions = {"taylor": 0.1, "statistical": 0.1, "baseline": 0.1}
        
        snapshot = watchdog.evaluate(weights, suppressions)
        
        assert snapshot.total_engines == 3
        assert snapshot.active_engines == 3
        assert snapshot.suppressed_engines == 0
        assert snapshot.min_weight == 0.3
        assert snapshot.max_suppression == 0.1
        assert snapshot.health == EnsembleHealth.HEALTHY
        assert snapshot.reason is not None
        assert isinstance(snapshot.reason, str)

    def test_single_engine_ensemble(self):
        """Single engine ensemble edge case."""
        watchdog = EnsembleWatchdog()
        weights = {"taylor": 0.5}
        suppressions = {"taylor": 0.1}
        
        snapshot = watchdog.evaluate(weights, suppressions)
        
        assert snapshot.total_engines == 1
        assert snapshot.active_engines == 1
        assert snapshot.health == EnsembleHealth.HEALTHY

    def test_empty_ensemble(self):
        """Empty ensemble edge case."""
        watchdog = EnsembleWatchdog()
        weights = {}
        suppressions = {}
        
        snapshot = watchdog.evaluate(weights, suppressions)
        
        assert snapshot.total_engines == 0
        assert snapshot.active_engines == 0
        assert snapshot.health == EnsembleHealth.COLLAPSED
        assert snapshot.recovery_recommended is True
