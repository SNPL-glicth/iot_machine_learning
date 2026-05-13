"""Tests for SEVERO-4: PhaseTimer timeout enforcement."""

import time

import pytest

from iot_machine_learning.infrastructure.ml.cognitive.orchestration.phase_timer import (
    PhaseTimer,
)


class TestPhaseTimer:
    """Test PhaseTimer (SEVERO-4)."""
    
    def test_constructor_validation(self):
        """Constructor validates parameters."""
        # Valid
        timer = PhaseTimer(total_budget_ms=1000.0)
        assert timer._total_budget_ms == 1000.0
        
        # Invalid budget
        with pytest.raises(ValueError, match="total_budget_ms must be > 0"):
            PhaseTimer(total_budget_ms=0.0)
        
        # Invalid weights (don't sum to 1.0)
        with pytest.raises(ValueError, match="Phase weights must sum to ~1.0"):
            PhaseTimer(
                total_budget_ms=1000.0,
                phase_weights={"phase1": 0.5, "phase2": 0.3},  # Sum = 0.8
            )
    
    def test_start_timer(self):
        """start() initializes timer."""
        timer = PhaseTimer(total_budget_ms=1000.0)
        
        assert timer._start_time is None
        
        timer.start()
        
        assert timer._start_time is not None
        assert timer.get_elapsed_ms() >= 0.0
        assert timer.get_remaining_ms() <= 1000.0
    
    def test_start_phase_without_start_raises(self):
        """start_phase() raises if timer not started."""
        timer = PhaseTimer(total_budget_ms=1000.0)
        
        with pytest.raises(RuntimeError, match="Timer not started"):
            timer.start_phase("perceive")
    
    def test_start_phase_sufficient_time(self):
        """start_phase() succeeds when sufficient time remains."""
        timer = PhaseTimer(total_budget_ms=1000.0)
        timer.start()
        
        # Should not raise
        timer.start_phase("perceive")
        assert timer._current_phase == "perceive"
    
    def test_start_phase_insufficient_time_raises(self):
        """start_phase() raises TimeoutError when insufficient time (SEVERO-4)."""
        timer = PhaseTimer(total_budget_ms=100.0)
        timer.start()
        
        # Sleep to consume most of budget
        time.sleep(0.095)  # 95ms
        
        # predict phase has 35% weight = 35ms budget
        # Remaining ~5ms < 35ms * 0.5 = 17.5ms → should raise
        with pytest.raises(TimeoutError, match="Insufficient time for phase 'predict'"):
            timer.start_phase("predict")
    
    def test_get_phase_budget_ms(self):
        """get_phase_budget_ms() returns correct budget."""
        timer = PhaseTimer(total_budget_ms=1000.0)
        
        # predict has 35% weight
        assert timer.get_phase_budget_ms("predict") == 350.0
        
        # perceive has 5% weight
        assert timer.get_phase_budget_ms("perceive") == 50.0
        
        # Unknown phase defaults to 5%
        assert timer.get_phase_budget_ms("unknown_phase") == 50.0
    
    def test_get_remaining_ms(self):
        """get_remaining_ms() returns correct remaining time."""
        timer = PhaseTimer(total_budget_ms=1000.0)
        
        # Before start
        assert timer.get_remaining_ms() == 1000.0
        
        # After start
        timer.start()
        time.sleep(0.01)  # 10ms
        
        remaining = timer.get_remaining_ms()
        assert 980.0 <= remaining <= 1000.0  # Allow some timing variance
    
    def test_get_elapsed_ms(self):
        """get_elapsed_ms() returns correct elapsed time."""
        timer = PhaseTimer(total_budget_ms=1000.0)
        
        # Before start
        assert timer.get_elapsed_ms() == 0.0
        
        # After start
        timer.start()
        time.sleep(0.01)  # 10ms
        
        elapsed = timer.get_elapsed_ms()
        assert 5.0 <= elapsed <= 20.0  # Allow timing variance
    
    def test_is_over_budget(self):
        """is_over_budget() detects budget exceeded."""
        timer = PhaseTimer(total_budget_ms=50.0)
        
        # Before start
        assert not timer.is_over_budget()
        
        # After start, within budget
        timer.start()
        assert not timer.is_over_budget()
        
        # After exceeding budget
        time.sleep(0.06)  # 60ms > 50ms budget
        assert timer.is_over_budget()
    
    def test_get_metrics(self):
        """get_metrics() returns timer state."""
        timer = PhaseTimer(total_budget_ms=1000.0)
        timer.start()
        timer.start_phase("perceive")
        
        metrics = timer.get_metrics()
        
        assert metrics["total_budget_ms"] == 1000.0
        assert metrics["elapsed_ms"] >= 0.0
        assert metrics["remaining_ms"] <= 1000.0
        assert metrics["current_phase"] == "perceive"
        assert isinstance(metrics["is_over_budget"], bool)
    
    def test_custom_phase_weights(self):
        """Custom phase weights are respected."""
        custom_weights = {
            "phase1": 0.6,
            "phase2": 0.4,
        }
        
        timer = PhaseTimer(total_budget_ms=1000.0, phase_weights=custom_weights)
        
        assert timer.get_phase_budget_ms("phase1") == 600.0
        assert timer.get_phase_budget_ms("phase2") == 400.0
    
    def test_default_phase_weights_sum_to_one(self):
        """Default phase weights sum to 1.0."""
        weight_sum = sum(PhaseTimer.DEFAULT_PHASE_WEIGHTS.values())
        assert 0.99 <= weight_sum <= 1.01
