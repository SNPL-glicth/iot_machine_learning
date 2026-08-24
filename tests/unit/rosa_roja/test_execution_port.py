"""Tests for ExecutionPort protocol and implementations."""

from __future__ import annotations

import pytest
from typing import Dict, Any, Optional
from unittest.mock import Mock

from core.orchestration.rosa_roja.ports.execution_port import ExecutionPort
from core.orchestration.rosa_roja.domain.execution import ExecutionPlan, ActionEnvelope
from core.orchestration.rosa_roja.domain.trajectory import Trajectory, TerminalState
from core.orchestration.rosa_roja.domain.movement import Movement, RhythmSignature
import numpy as np


class MockExecutionHandler:
    """Mock execution handler implementing ExecutionPort for testing."""
    
    def __init__(self):
        self.dispatched_plans = []
        self.emergency_flushes = []
        self.should_fail = False
    
    def dispatch_execution(self, plan: ExecutionPlan) -> bool:
        self.dispatched_plans.append(plan)
        if self.should_fail:
            return False
        return True
    
    def trigger_emergency_flush(self, reason: str) -> None:
        self.emergency_flushes.append(reason)


class TestExecutionPortProtocol:
    """Tests verifying ExecutionPort protocol compliance."""
    
    def test_mock_handler_implements_protocol(self):
        """MockExecutionHandler should satisfy ExecutionPort protocol."""
        handler = MockExecutionHandler()
        assert isinstance(handler, ExecutionPort)
    
    def test_dispatch_execution_returns_bool(self):
        """dispatch_execution should return boolean success indicator."""
        handler = MockExecutionHandler()
        plan = ExecutionPlan.HOLD("test")
        result = handler.dispatch_execution(plan)
        assert isinstance(result, bool)
    
    def test_trigger_emergency_flush_accepts_reason(self):
        """trigger_emergency_flush should accept string reason."""
        handler = MockExecutionHandler()
        handler.trigger_emergency_flush("test_reason")
        assert "test_reason" in handler.emergency_flushes
    
    def test_protocol_enforces_signature(self):
        """Protocol should enforce method signatures via static checking."""
        # This is verified by type checker (mypy/pyright)
        # Runtime check: methods exist and are callable
        handler = MockExecutionHandler()
        assert callable(getattr(handler, "dispatch_execution", None))
        assert callable(getattr(handler, "trigger_emergency_flush", None))


class TestExecutionPlanWithEnvelope:
    """Tests for ExecutionPlan with ActionEnvelope."""
    
    def create_test_trajectory(self, length: int = 3) -> Trajectory:
        movements = []
        for i in range(length):
            delta_state = np.array([float(i), 0.0])
            rhythm = RhythmSignature(1.0, 0.0, 0.0, 0.0, 0.1)
            movement = Movement(delta_state, 1.0, 1.0, np.array([1.0, 0.0]), rhythm, 1.0, float(i))
            movements.append(movement)
        return Trajectory(
            movements=tuple(movements),
            coherence_score=0.8,
            invalidation_step=None,
            terminal_state=TerminalState(
                state_vector=movements[-1].delta_state,
                step_index=len(movements) - 1,
                confidence=0.8,
            ),
        )
    
    def test_execute_plan_contains_envelope(self):
        """EXECUTE plan should contain ActionEnvelope with parameters."""
        traj = self.create_test_trajectory()
        envelope = ActionEnvelope(
            magnitude=0.5,
            bounds={"stop_pct": 0.02, "target_pct": 0.05},
            max_steps=15,
            metadata={"test": True}
        )
        plan = ExecutionPlan.EXECUTE(traj, 0.8, envelope, invalidation_step=3)
        
        assert plan.action == "EXECUTE"
        assert plan.envelope is not None
        assert plan.envelope.magnitude == 0.5
        assert plan.envelope.bounds["stop_pct"] == 0.02
        assert plan.envelope.max_steps == 15
    
    def test_hold_plan_has_no_envelope(self):
        """HOLD plan should have None envelope."""
        plan = ExecutionPlan.HOLD("test_reason")
        assert plan.action == "HOLD"
        assert plan.envelope is None
    
    def test_emergency_flush_plan_has_no_envelope(self):
        """EMERGENCY_FLUSH plan should have None envelope."""
        plan = ExecutionPlan.EMERGENCY_FLUSH("test_reason")
        assert plan.action == "EMERGENCY_FLUSH"
        assert plan.envelope is None
        assert plan.regime_alert is True


class TestActionEnvelope:
    """Tests for ActionEnvelope domain model."""
    
    def test_envelope_creation(self):
        """ActionEnvelope should store all parameters."""
        envelope = ActionEnvelope(
            magnitude=0.75,
            bounds={"stop": 0.02, "target": 0.05, "custom": "value"},
            max_steps=20,
            metadata={"source": "test"}
        )
        
        assert envelope.magnitude == 0.75
        assert envelope.bounds["stop"] == 0.02
        assert envelope.max_steps == 20
        assert envelope.metadata["source"] == "test"
    
    def test_envelope_bounds_are_arbitrary_dict(self):
        """Envelope bounds should accept arbitrary key-value pairs."""
        envelope = ActionEnvelope(
            magnitude=1.0,
            bounds={"temperature_limit": 80.0, "pressure_max": 10.5, "mode": "auto"},
            max_steps=100,
            metadata={}
        )
        
        assert envelope.bounds["temperature_limit"] == 80.0
        assert envelope.bounds["pressure_max"] == 10.5
        assert envelope.bounds["mode"] == "auto"
    
    def test_envelope_immutability(self):
        """ActionEnvelope should be frozen (immutable)."""
        envelope = ActionEnvelope(0.5, {}, 10, {})
        
        with pytest.raises(Exception):
            envelope.magnitude = 1.0


class TestMockHandlerIntegration:
    """Integration tests with mock execution handler."""
    
    def create_test_trajectory(self, length: int = 3) -> Trajectory:
        movements = []
        for i in range(length):
            delta_state = np.array([float(i), 0.0])
            rhythm = RhythmSignature(1.0, 0.0, 0.0, 0.0, 0.1)
            movement = Movement(delta_state, 1.0, 1.0, np.array([1.0, 0.0]), rhythm, 1.0, float(i))
            movements.append(movement)
        return Trajectory(
            movements=tuple(movements),
            coherence_score=0.8,
            invalidation_step=None,
            terminal_state=TerminalState(
                state_vector=movements[-1].delta_state,
                step_index=len(movements) - 1,
                confidence=0.8,
            ),
        )
    
    def test_handler_receives_execute_plan(self):
        """Handler should receive and store EXECUTE plans."""
        handler = MockExecutionHandler()
        traj = self.create_test_trajectory()
        envelope = ActionEnvelope(0.5, {"stop_pct": 0.02}, 15, {})
        plan = ExecutionPlan.EXECUTE(traj, 0.8, envelope)
        
        result = handler.dispatch_execution(plan)
        
        assert result is True
        assert len(handler.dispatched_plans) == 1
        assert handler.dispatched_plans[0] is plan
    
    def test_handler_receives_hold_plan(self):
        """Handler should receive and store HOLD plans."""
        handler = MockExecutionHandler()
        plan = ExecutionPlan.HOLD("Insufficient history")
        
        result = handler.dispatch_execution(plan)
        
        assert result is True
        assert len(handler.dispatched_plans) == 1
        assert handler.dispatched_plans[0].action == "HOLD"
    
    def test_handler_emergency_flush(self):
        """Handler should trigger emergency flush."""
        handler = MockExecutionHandler()
        
        handler.trigger_emergency_flush("Module 1 outlier")
        
        assert len(handler.emergency_flushes) == 1
        assert "Module 1 outlier" in handler.emergency_flushes[0]
    
    def test_handler_returns_false_on_failure(self):
        """Handler should return False when execution fails."""
        handler = MockExecutionHandler()
        handler.should_fail = True
        plan = ExecutionPlan.HOLD("test")
        
        result = handler.dispatch_execution(plan)
        
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])