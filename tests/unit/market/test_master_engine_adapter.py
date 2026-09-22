"""Unit tests for Zephyr Master Engine Adapter module.

Validates the 4 Golden Rules:
1. Total Sovereignty of MasterEquationOrchestrator
2. Strict <= 180 Lines Limit per file
3. Clean Code and strong typing
4. Zero Silent Errors and capital-protective failsafes
"""

from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from iot_machine_learning.infrastructure.adapters.market.zephyr.master_engine_adapter import (
    MasterEngineAdapter,
    MasterTelemetryPacker,
    OrderDirective,
    PlanTranslator,
    extract_decision_trace_metrics,
    resolve_plan_action_trigger,
)
from iot_machine_learning.domain.entities.rosa_roja.execution import (
    ActionEnvelope,
    ExecutionPlan,
)


def test_rule_1_sovereignty_translation():
    """Rule 1: Verify Master Equation dictate translates unambiguously into directives."""
    envelope = ActionEnvelope(
        magnitude=2.5,
        bounds={"target_pct": 0.04, "stop_pct": 0.02},
        max_steps=12,
        metadata={"decision_trace": {"lambda_t": 0.35, "phi_ritmo": 0.88}},
    )
    plan = ExecutionPlan(
        action=1.0,
        chosen_trajectory=MagicMock(),
        global_confidence=0.92,
        envelope=envelope,
        invalidation_step=None,
        regime_alert=False,
        veto_details={},
    )

    directive = PlanTranslator.translate(plan)

    assert isinstance(directive, OrderDirective)
    assert directive.action == "EXECUTE"
    assert directive.side == "buy"
    assert directive.trigger_code == 1
    assert directive.confidence == 0.92
    assert directive.magnitude == 2.5
    assert directive.is_executable is True
    assert directive.is_flush is False
    assert directive.is_hold is False


def test_rule_2_file_line_limits():
    """Rule 2: Verify all master engine adapter files and pipeline are <= 180 lines."""
    base_dir = Path(__file__).resolve().parents[3] / "infrastructure" / "adapters" / "market" / "zephyr"
    files_to_check = [
        base_dir / "execution" / "pipeline.py",
        base_dir / "master_engine_adapter" / "__init__.py",
        base_dir / "master_engine_adapter" / "adapter.py",
        base_dir / "master_engine_adapter" / "translator.py",
        base_dir / "master_engine_adapter" / "telemetry_packer.py",
    ]

    for fpath in files_to_check:
        assert fpath.exists(), f"File {fpath} does not exist"
        lines = fpath.read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 180, f"{fpath.name} exceeds 180 lines: {len(lines)} lines"


def test_rule_4_zero_silent_errors_failsafe_hold():
    """Rule 4: Master Equation crash with flat position returns failsafe HOLD without raising."""
    crashing_engine = MagicMock()
    crashing_engine.process_event.side_effect = RuntimeError("Numerical singularity in MoE")

    adapter = MasterEngineAdapter(crashing_engine, symbol="SPY")
    delta_s = np.zeros(10, dtype=np.float32)

    # Flat position -> safe action is HOLD
    plan = adapter.evaluate(delta_s, delta_time=0.1, current_position=0.0)

    assert isinstance(plan, ExecutionPlan)
    assert plan.action == "HOLD"
    assert plan.regime_alert is True
    assert "FAILSAFE_ZENIN_CRASH" in plan.veto_details["reason"]


def test_rule_4_zero_silent_errors_failsafe_flush():
    """Rule 4: Master Equation crash with open position returns failsafe EMERGENCY_FLUSH."""
    crashing_engine = MagicMock()
    crashing_engine.process_event.side_effect = ZeroDivisionError("division by zero in CVaR")

    adapter = MasterEngineAdapter(crashing_engine, symbol="SPY")
    delta_s = np.zeros(10, dtype=np.float32)

    # Open position -> safe action is EMERGENCY_FLUSH to protect capital
    plan = adapter.evaluate(delta_s, delta_time=0.1, current_position=15.0)

    assert isinstance(plan, ExecutionPlan)
    assert plan.action == "EMERGENCY_FLUSH"
    assert plan.regime_alert is True
    assert "FAILSAFE_ZENIN_CRASH" in plan.veto_details["reason"]


def test_master_telemetry_packer_weaviate_records():
    """Verify MasterTelemetryPacker constructs valid Weaviate payloads."""
    record = MasterTelemetryPacker.create_weaviate_telemetry_record(
        sym="SPY",
        mid=520.50,
        phi_moe=0.85,
        lambda_t=0.22,
        trigger_side="buy",
        trigger_code=1,
        reason="Trajectory_Coherence",
        scores={"cvar_t": 0.012, "certeza": 0.84},
    )

    assert record["symbol"] == "SPY"
    assert record["mid_price"] == 520.50
    assert record["action"] == "BUY:1"
    assert record["cvar"] == 0.012
    assert record["decision_rationale"]["certeza"] == 0.84


def test_evaluate_directive_bridge():
    """Verify MasterEngineAdapter.evaluate_directive returns both ExecutionPlan and OrderDirective."""
    mock_engine = MagicMock()
    mock_plan = ExecutionPlan(
        action=1.0,
        chosen_trajectory=MagicMock(),
        global_confidence=0.88,
        envelope=ActionEnvelope(magnitude=1.0, bounds={}, max_steps=10, metadata={}),
        invalidation_step=None,
        regime_alert=False,
        veto_details={},
    )
    mock_engine.process_event.return_value = mock_plan

    adapter = MasterEngineAdapter(mock_engine, symbol="SPY")
    delta_s = np.zeros(10, dtype=np.float32)

    plan, directive = adapter.evaluate_directive(delta_s, delta_time=0.1, current_position=0.0)

    assert plan is mock_plan
    assert isinstance(directive, OrderDirective)
    assert directive.action == "EXECUTE"
    assert directive.side == "buy"
    assert directive.confidence == 0.88


def test_orchestrator_default_active_mode():
    """Verify MasterEquationOrchestrator defaults to active institutional mode (shadow_mode=False)."""
    from iot_machine_learning.infrastructure.ml.master_engine import MasterEquationOrchestrator

    orch = MasterEquationOrchestrator(
        rosa_roja_engine=MagicMock(),
        risk_adapter=MagicMock(),
        temporal_adapter=MagicMock(),
    )
    assert orch._shadow_mode is False
