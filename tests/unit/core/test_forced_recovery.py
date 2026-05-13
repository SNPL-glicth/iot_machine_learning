"""Tests for ForcedRecoveryManager - Fase 11."""

import pytest

from core.ensemble.ensemble_watchdog import (
    EnsembleHealth,
    WatchdogSnapshot,
)
from core.ensemble.forced_recovery import (
    ForcedRecoveryManager,
    RecoveryResult,
    RecoveryStrategy,
)


class TestForcedRecoveryInit:
    """Tests for ForcedRecoveryManager initialization."""

    def test_init_default_params(self):
        """RecoveryManager initializes with default parameters."""
        recovery = ForcedRecoveryManager()
        assert recovery._soft_reduction_factor == 0.5
        assert recovery._min_recovery_weight == 0.1
        assert recovery._max_engines_to_recover == 2


class TestForcedRecoveryExecution:
    """Tests for recovery execution strategies."""

    def test_soft_recovery_reduces_suppression(self):
        """SOFT recovery reduces suppression by doubling low weights."""
        recovery = ForcedRecoveryManager()
        snapshot = WatchdogSnapshot(
            total_engines=3,
            active_engines=1,
            suppressed_engines=2,
            min_weight=0.01,
            max_suppression=0.95,
            health=EnsembleHealth.CRITICAL,
            recovery_recommended=True,
            reason="test",
        )
        weights = {"taylor": 0.01, "statistical": 0.01, "baseline": 0.98}
        suppressions = {"taylor": 0.95, "statistical": 0.95, "baseline": 0.1}
        
        result = recovery.execute(snapshot, weights, suppressions)
        
        assert result.strategy_used == RecoveryStrategy.SOFT
        assert result.success is True
        assert len(result.engines_recovered) > 0
        assert "taylor" in result.weight_adjustments or "statistical" in result.weight_adjustments

    def test_hard_recovery_resets_all(self):
        """HARD recovery resets all weights to equal distribution."""
        recovery = ForcedRecoveryManager()
        snapshot = WatchdogSnapshot(
            total_engines=3,
            active_engines=0,
            suppressed_engines=3,
            min_weight=0.01,
            max_suppression=0.95,
            health=EnsembleHealth.COLLAPSED,
            recovery_recommended=True,
            reason="test",
        )
        weights = {"taylor": 0.01, "statistical": 0.01, "baseline": 0.01}
        suppressions = {"taylor": 0.95, "statistical": 0.95, "baseline": 0.95}
        
        result = recovery.execute(snapshot, weights, suppressions)
        
        assert result.strategy_used == RecoveryStrategy.HARD
        assert result.success is True
        assert len(result.engines_recovered) == 3
        assert all(w == pytest.approx(1.0/3) for w in result.weight_adjustments.values())

    def test_selective_recovery_picks_best_engines(self):
        """SELECTIVE recovery picks engines with lowest errors."""
        recovery = ForcedRecoveryManager()
        snapshot = WatchdogSnapshot(
            total_engines=3,
            active_engines=1,
            suppressed_engines=2,
            min_weight=0.01,
            max_suppression=0.95,
            health=EnsembleHealth.CRITICAL,
            recovery_recommended=True,
            reason="test",
        )
        weights = {"taylor": 0.01, "statistical": 0.01, "baseline": 0.98}
        suppressions = {"taylor": 0.95, "statistical": 0.95, "baseline": 0.1}
        engine_errors = {"taylor": 0.3, "statistical": 0.5, "baseline": 0.1}
        
        result = recovery.execute(snapshot, weights, suppressions, engine_errors)
        
        assert result.strategy_used == RecoveryStrategy.SELECTIVE
        assert result.success is True
        assert len(result.engines_recovered) <= 2  # max_engines_to_recover

    def test_collapsed_triggers_hard(self):
        """COLLAPSED health triggers HARD recovery."""
        recovery = ForcedRecoveryManager()
        snapshot = WatchdogSnapshot(
            total_engines=3,
            active_engines=0,
            suppressed_engines=3,
            min_weight=0.01,
            max_suppression=0.95,
            health=EnsembleHealth.COLLAPSED,
            recovery_recommended=True,
            reason="test",
        )
        weights = {"taylor": 0.01, "statistical": 0.01, "baseline": 0.01}
        suppressions = {"taylor": 0.95, "statistical": 0.95, "baseline": 0.95}
        
        result = recovery.execute(snapshot, weights, suppressions)
        
        assert result.strategy_used == RecoveryStrategy.HARD

    def test_critical_triggers_soft(self):
        """CRITICAL health without errors triggers SOFT recovery."""
        recovery = ForcedRecoveryManager()
        snapshot = WatchdogSnapshot(
            total_engines=3,
            active_engines=1,
            suppressed_engines=2,
            min_weight=0.01,
            max_suppression=0.95,
            health=EnsembleHealth.CRITICAL,
            recovery_recommended=True,
            reason="test",
        )
        weights = {"taylor": 0.01, "statistical": 0.01, "baseline": 0.98}
        suppressions = {"taylor": 0.95, "statistical": 0.95, "baseline": 0.1}
        
        result = recovery.execute(snapshot, weights, suppressions)
        
        assert result.strategy_used == RecoveryStrategy.SOFT

    def test_recovery_result_fields(self):
        """RecoveryResult has all required fields."""
        recovery = ForcedRecoveryManager()
        snapshot = WatchdogSnapshot(
            total_engines=3,
            active_engines=0,
            suppressed_engines=3,
            min_weight=0.01,
            max_suppression=0.95,
            health=EnsembleHealth.COLLAPSED,
            recovery_recommended=True,
            reason="test",
        )
        weights = {"taylor": 0.01, "statistical": 0.01, "baseline": 0.01}
        suppressions = {"taylor": 0.95, "statistical": 0.95, "baseline": 0.95}
        
        result = recovery.execute(snapshot, weights, suppressions)
        
        assert hasattr(result, "strategy_used")
        assert hasattr(result, "engines_recovered")
        assert hasattr(result, "weight_adjustments")
        assert hasattr(result, "reason")
        assert hasattr(result, "success")
        assert isinstance(result.engines_recovered, list)
        assert isinstance(result.weight_adjustments, dict)

    def test_weight_adjustments_sum_to_one(self):
        """Weight adjustments from HARD recovery sum to 1.0."""
        recovery = ForcedRecoveryManager()
        snapshot = WatchdogSnapshot(
            total_engines=3,
            active_engines=0,
            suppressed_engines=3,
            min_weight=0.01,
            max_suppression=0.95,
            health=EnsembleHealth.COLLAPSED,
            recovery_recommended=True,
            reason="test",
        )
        weights = {"taylor": 0.01, "statistical": 0.01, "baseline": 0.01}
        suppressions = {"taylor": 0.95, "statistical": 0.95, "baseline": 0.95}
        
        result = recovery.execute(snapshot, weights, suppressions)
        
        total = sum(result.weight_adjustments.values())
        assert total == pytest.approx(1.0)

    def test_min_recovery_weight_respected(self):
        """Recovery respects min_recovery_weight threshold."""
        recovery = ForcedRecoveryManager(min_recovery_weight=0.2)
        snapshot = WatchdogSnapshot(
            total_engines=3,
            active_engines=1,
            suppressed_engines=2,
            min_weight=0.01,
            max_suppression=0.95,
            health=EnsembleHealth.CRITICAL,
            recovery_recommended=True,
            reason="test",
        )
        weights = {"taylor": 0.01, "statistical": 0.01, "baseline": 0.98}
        suppressions = {"taylor": 0.95, "statistical": 0.95, "baseline": 0.1}
        
        result = recovery.execute(snapshot, weights, suppressions)
        
        for weight in result.weight_adjustments.values():
            assert weight >= 0.2

    def test_empty_ensemble_graceful(self):
        """Empty ensemble handled gracefully."""
        recovery = ForcedRecoveryManager()
        snapshot = WatchdogSnapshot(
            total_engines=0,
            active_engines=0,
            suppressed_engines=0,
            min_weight=0.0,
            max_suppression=0.0,
            health=EnsembleHealth.COLLAPSED,
            recovery_recommended=True,
            reason="test",
        )
        weights = {}
        suppressions = {}
        
        result = recovery.execute(snapshot, weights, suppressions)
        
        assert result.success is False
        assert len(result.engines_recovered) == 0
