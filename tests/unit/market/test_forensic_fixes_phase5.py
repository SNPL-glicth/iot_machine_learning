"""Unit tests for Forensic Fixes Phase 5:
1. Trajectory tracker desynchronization reset (cancel_active_trajectory on can_execute or dispatch failure).
2. Emergency flush on flat positions does not lock out symbols with 45s cooldown.
3. Segregated asset clusters allow concurrent execution in multi-asset baskets.
4. Order sizing correctly respects _max_qty upper limit.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from iot_machine_learning.infrastructure.adapters.market.zephyr.config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.zephyr.models import LiveBotState
from iot_machine_learning.infrastructure.adapters.market.zephyr.risk.portfolio_risk_manager import (
    PortfolioRiskConfig,
    PortfolioRiskManager,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja.market_handler import (
    RosaRojaMarketExecutionHandler,
)
from infrastructure.ml.engines.rosa_roja.algorithms.domain.execution import (
    ActionEnvelope,
    ExecutionPlan,
)
from infrastructure.ml.engines.rosa_roja.algorithms.domain.state_machine import (
    PipelineState,
    StateMachine,
    TrackingState,
)
from infrastructure.ml.engines.rosa_roja.algorithms.domain.trajectory_tracker import (
    TrajectoryTracker,
)
from infrastructure.ml.engines.rosa_roja.algorithms.engine import RosaRojaEngine


def test_fix1_cancel_active_trajectory_resets_tracker_and_state_machine():
    """Verifica que cancel_active_trajectory limpia el tracker y regresa la máquina de estados a GENERATING."""
    state_machine = StateMachine()
    tracker = TrajectoryTracker(max_direction_dev_deg=120.0, max_velocity_rel_err=5.0)

    # Crear motor con tracker y state_machine
    engine = RosaRojaEngine(
        ingestion_filter=MagicMock(),
        rhythm_generator=MagicMock(),
        moe_gating=MagicMock(),
        expert_jury=[],
        shadow_experts=[],
        drift_sensors=[],
    )
    engine._state_machine = state_machine
    engine._tracker = tracker

    # Simular inicio de trayectoria con mock
    mock_traj = MagicMock(invalidation_step=5)
    tracker.set_active_trajectory(mock_traj, start_step=1)
    state_machine.on_trajectory_start("traj_1", invalidation_step=5)

    assert tracker.has_active_trajectory is True
    assert state_machine.state.tracking_state == TrackingState.TRACKING
    assert state_machine.state.pipeline_state == PipelineState.ACTIVE_TRACKING

    # Ejecutar cancelación quirúrgica
    engine.cancel_active_trajectory()

    assert tracker.has_active_trajectory is False
    assert tracker.active_trajectory is None
    assert state_machine.state.tracking_state == TrackingState.EXPIRED
    assert state_machine.state.pipeline_state == PipelineState.GENERATING


@pytest.mark.asyncio
async def test_fix1_live_runner_cancels_trajectory_when_can_execute_vetoes():
    """Verifica que si can_execute veta una orden EXECUTE, el LiveBotRunner cancela la trayectoria en el motor."""
    from iot_machine_learning.infrastructure.adapters.market.zephyr.runners import LiveBotRunner

    config = LiveBotConfig(symbol="SPY", phi_moe_threshold=0.40)
    runner = LiveBotRunner(config=config, feed=MagicMock())

    # Motor mock
    mock_engine = MagicMock()
    mock_engine.process_event.return_value = ExecutionPlan.EXECUTE(
        trajectory=MagicMock(),
        confidence=0.85,
        envelope=MagicMock(),
        invalidation_step=5,
    )
    mock_engine.cancel_active_trajectory = MagicMock()

    runner._engine = mock_engine
    runner._symbol_engines = {"SPY": mock_engine}
    runner._feature_extractor = MagicMock()
    runner._feature_extractor.process.return_value = (np.array([1.0]), 0.1)

    # Forzar que _can_execute retorne False (ej. veto de riesgo)
    runner._can_execute = MagicMock(return_value=False)

    obs = MagicMock()
    obs.symbol = "SPY"
    await runner._process_observation(obs)

    # Verificar que cancel_active_trajectory fue llamado para evitar orden fantasma
    mock_engine.cancel_active_trajectory.assert_called_once()


@pytest.mark.asyncio
async def test_fix2_emergency_flush_on_flat_position_does_not_arm_cooldown():
    """Verifica que un EMERGENCY_FLUSH en un símbolo plano (pos=0) no impone el cooldown de 45s."""
    broker = AsyncMock()
    broker.cancel_order = AsyncMock()
    broker.cancel_all_orders = AsyncMock(return_value=0)
    broker.get_position = AsyncMock(return_value={"qty": "0.0"})

    state = LiveBotState()
    handler = RosaRojaMarketExecutionHandler(
        broker_client=broker,
        account_equity=100000.0,
        symbol="SPY",
        state=state,
        lot_size=1.0,
    )

    assert state.is_symbol_closing("SPY") is False

    # Disparar emergency flush en posición plana
    await handler.trigger_emergency_flush(reason="Noise_Outlier_Test", symbol="SPY")

    # Debe cancelar órdenes en broker, pero NO debe bloquear al símbolo con 45s cooldown
    assert state.is_symbol_closing("SPY") is False
    assert state.is_closing.get("SPY", False) is False
    assert "SPY" not in state.close_confirmed_at


def test_fix3_segregated_clusters_allow_concurrent_basket_positions():
    """Verifica que con clusters segregados (SPY vs QQQ), ambos pueden abrir cortos hasta max_positions."""
    segregated_clusters = {
        "INDEX_BROAD": {"SPY", "VOO", "IVV"},
        "INDEX_TECH": {"QQQ", "TQQQ", "SQQQ"},
        "EQUITY_TECH_MEGA": {"AAPL", "MSFT"},
        "EQUITY_SEMIS": {"NVDA", "AMD"},
    }
    risk_cfg = PortfolioRiskConfig(max_cluster_positions=1)
    mgr = PortfolioRiskManager(initial_equity=100000.0, config=risk_cfg, clusters=segregated_clusters)

    # QQQ ya tiene posición corta activa
    open_positions = {"QQQ": -14.0, "SPY": 0.0, "AAPL": 0.0, "NVDA": 0.0}

    # Intentar abrir corto en SPY (cluster INDEX_BROAD distinto a INDEX_TECH) -> PERMITIDO
    allowed_spy, reason_spy = mgr.check_correlation_guardrail("SPY", "sell", open_positions)
    assert allowed_spy is True
    assert reason_spy == ""

    # Intentar abrir corto en AAPL (cluster EQUITY_TECH_MEGA) -> PERMITIDO
    allowed_aapl, reason_aapl = mgr.check_correlation_guardrail("AAPL", "sell", open_positions)
    assert allowed_aapl is True
    assert reason_aapl == ""

    # Intentar abrir otro corto en TQQQ (mismo cluster INDEX_TECH que QQQ) -> VETADO
    allowed_tqqq, reason_tqqq = mgr.check_correlation_guardrail("TQQQ", "sell", open_positions)
    assert allowed_tqqq is False
    assert "Correlation Guardrail VETO" in reason_tqqq
    assert "QQQ(-14.0)" in reason_tqqq


@pytest.mark.asyncio
async def test_fix4_order_sizing_strictly_capped_at_max_qty():
    """Verifica que dispatch_execution respeta el límite superior _max_qty."""
    broker = AsyncMock()
    broker.submit_order = AsyncMock(return_value=MagicMock(id="test_ord_1"))

    handler = RosaRojaMarketExecutionHandler(
        broker_client=broker,
        account_equity=1000000.0,  # 1 millón para forzar notional enorme
        symbol="SPY",
        max_position_pct=0.50,
        lot_size=1.0,
        max_qty=15.0,  # Límite máximo estricto: 15 acciones
    )
    handler._get_reference_price = MagicMock(return_value=100.0)

    mock_traj = MagicMock(terminal_state=MagicMock(state_vector=[105.0]), length=5)
    plan = ExecutionPlan.EXECUTE(
        trajectory=mock_traj,
        confidence=0.9,
        envelope=ActionEnvelope(magnitude=0.5, bounds={}, max_steps=5, metadata={}),
        invalidation_step=5,
    )

    success = await handler.dispatch_execution(plan, symbol="SPY")
    assert success is True

    # Verificar cantidad enviada a submit_order
    call_args = broker.submit_order.call_args[1]
    assert call_args["qty"] == 15.0  # Capped at max_lot_size
