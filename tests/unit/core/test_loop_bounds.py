"""Tests for LoopBoundsMonitor - Fase 11."""

import pytest

from core.ensemble.loop_bounds import (
    LoopBoundsConfig,
    LoopBoundsMonitor,
)


class TestLoopBoundsMonitorInit:
    """Tests for LoopBoundsMonitor initialization."""

    def test_init_default_config(self):
        """Monitor initializes with default config."""
        monitor = LoopBoundsMonitor()
        assert monitor._config.max_contamination_rate_of_change == 0.1
        assert monitor._config.contamination_cooldown_cycles == 5
        assert monitor._config.min_ensemble_mean_weight == 0.1
        assert monitor._config.max_weight_drop_per_cycle == 0.3
        assert monitor._config.max_simultaneous_suppressed == 2
        assert monitor._config.max_total_suppression == 0.7

    def test_custom_config_respected(self):
        """Custom config is respected."""
        config = LoopBoundsConfig(
            max_contamination_rate_of_change=0.2,
            contamination_cooldown_cycles=10,
        )
        monitor = LoopBoundsMonitor(config)
        assert monitor._config.max_contamination_rate_of_change == 0.2
        assert monitor._config.contamination_cooldown_cycles == 10


class TestContaminationBounds:
    """Tests for contamination update bounds checking."""

    def test_contamination_update_safe(self):
        """Safe contamination update passes check."""
        monitor = LoopBoundsMonitor()
        # 0.005 -> 0.0055 is 10% change (within 0.1 threshold)
        assert monitor.check_contamination_update(0.005, 0.0055) is True

    def test_contamination_update_too_fast(self):
        """Update exceeding rate_of_change is blocked."""
        monitor = LoopBoundsMonitor()
        assert monitor.check_contamination_update(0.005, 0.015) is False

    def test_contamination_cooldown_blocks(self):
        """Update during cooldown is blocked."""
        monitor = LoopBoundsMonitor()
        monitor.increment_contamination_cycle()
        monitor.increment_contamination_cycle()
        monitor.increment_contamination_cycle()
        # Still in cooldown (need 5 cycles)
        assert monitor.check_contamination_update(0.005, 0.006) is False


class TestWeightStateBounds:
    """Tests for weight state bounds checking."""

    def test_weight_state_healthy(self):
        """Healthy weight state passes check."""
        monitor = LoopBoundsMonitor()
        weights = {"taylor": 0.4, "statistical": 0.3, "baseline": 0.3}
        assert monitor.check_weight_state(weights) is True

    def test_weight_state_too_low(self):
        """Mean weight below min is blocked."""
        monitor = LoopBoundsMonitor()
        weights = {"taylor": 0.01, "statistical": 0.01, "baseline": 0.01}
        assert monitor.check_weight_state(weights) is False


class TestSuppressionStateBounds:
    """Tests for suppression state bounds checking."""

    def test_suppression_state_ok(self):
        """Suppression within bounds passes check."""
        monitor = LoopBoundsMonitor()
        suppressions = {"taylor": 0.1, "statistical": 0.1, "baseline": 0.1}
        assert monitor.check_suppression_state(suppressions) is True

    def test_suppression_state_too_many(self):
        """Too many engines suppressed simultaneously is blocked."""
        monitor = LoopBoundsMonitor()
        suppressions = {"taylor": 0.9, "statistical": 0.9, "baseline": 0.9}
        assert monitor.check_suppression_state(suppressions) is False


class TestLoopHealthSummary:
    """Tests for loop health summary."""

    def test_loop_health_summary_structure(self):
        """Health summary has correct structure."""
        monitor = LoopBoundsMonitor()
        summary = monitor.get_loop_health_summary()
        
        assert "contamination_cycles" in summary
        assert "last_contamination_value" in summary
        assert "config" in summary
        assert "max_contamination_rate_of_change" in summary["config"]
