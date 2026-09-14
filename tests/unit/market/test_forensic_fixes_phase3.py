"""Unit and integration tests for Phase 3 risk management fixes (Fix 1 and Fix 2)."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from infrastructure.adapters.market.live_config import LiveBotConfig
from infrastructure.adapters.market.live_runner_execution import can_execute
from infrastructure.adapters.market.live_runner_models import LiveBotState
from infrastructure.adapters.market.portfolio_risk_manager import (
    PortfolioRiskConfig,
    PortfolioRiskManager,
)
from infrastructure.adapters.market.rosa_roja_market_handler import RosaRojaMarketExecutionHandler
from infrastructure.ml.engines.rosa_roja.algorithms.domain.execution import ExecutionPlan


# ============================================================================
# FIX 1: Límite de Pérdida Diaria Máxima Absoluta
# ============================================================================

def test_fix1_circuit_breaker_trips_on_negative_start_without_prior_profit():
    """Fix 1: Verifies circuit breaker trips when account starts day losing without prior profit."""
    cfg = PortfolioRiskConfig(
        profit_lock_trigger_usd=10.0,
        max_giveback_pct=0.25,
        max_daily_loss_usd=50.0,
    )
    # Account starts at $10,000.00
    mgr = PortfolioRiskManager(initial_equity=10000.0, config=cfg)

    # 1. Loss of $20 -> Equity $9,980.00 (not tripped yet)
    tripped, reason = mgr.update_equity(9980.0)
    assert tripped is False
    assert reason == ""

    # 2. Cumulative loss reaches $50 -> Equity $9,950.00 (MUST TRIP!)
    tripped, reason = mgr.update_equity(9950.0)
    assert tripped is True
    assert "CRITICAL: Maximum Daily Loss Limit Reached!" in reason
    assert "Loss: -$50.00 >= Limit: -$50.00" in reason
    assert mgr.circuit_breaker_tripped is True

    # 3. Subsequent checks stay tripped
    tripped_again, _ = mgr.update_equity(9940.0)
    assert tripped_again is True


def test_fix1_circuit_breaker_blocks_can_execute_globally():
    """Fix 1: Verifies that when daily circuit breaker trips, all entry orders on all symbols are vetoed."""
    state = LiveBotState()
    state.portfolio_circuit_breaker_tripped = True
    config = LiveBotConfig(symbol="SPY")

    plan = ExecutionPlan.HOLD("test")
    # Non-EXECUTE action passes
    assert can_execute(plan, config, state, current_price=500.0, symbol="SPY") is True

    # EXECUTE action is blocked globally across all symbols
    exec_plan = MagicMock(action="EXECUTE")
    assert can_execute(exec_plan, config, state, current_price=500.0, symbol="SPY") is False
    assert can_execute(exec_plan, config, state, current_price=400.0, symbol="QQQ") is False


def test_fix1_preserves_profit_lock_giveback():
    """Fix 1: Verifies high-water mark profit lock still functions if account was in profit."""
    cfg = PortfolioRiskConfig(
        profit_lock_trigger_usd=10.0,
        max_giveback_pct=0.25,
        max_daily_loss_usd=50.0,
    )
    mgr = PortfolioRiskManager(initial_equity=10000.0, config=cfg)

    # Peak profit reaches +$20 -> Equity $10,020.00 (giveback allowed: $5 -> floor $10,015.00)
    mgr.update_equity(10020.0)
    assert mgr.is_profit_locked is True

    # Drop to $10,014.00 (below floor $10,015.00) -> trips giveback breaker
    tripped, reason = mgr.update_equity(10014.0)
    assert tripped is True
    assert "Portfolio Circuit Breaker Triggered: Peak profit was $20.00" in reason


# ============================================================================
# FIX 2: Contador de Pérdidas Consecutivas y Cooldown de Enfriamiento
# ============================================================================

def test_fix2_consecutive_loss_counter_and_cooling_cooldown():
    """Fix 2: Verifies consecutive losses trigger 15-minute cooling cooldown and win resets."""
    state = LiveBotState()
    state.last_phi_moe = 0.8  # Satisfies config.phi_moe_threshold
    config = LiveBotConfig(symbol="SPY")
    exec_plan = MagicMock(action="EXECUTE")

    # Initial state: can execute
    assert can_execute(exec_plan, config, state, current_price=500.0, symbol="SPY") is True

    # Trade 1: Loss of -$10 -> count=1 (threshold=2)
    cooling = state.record_trade_outcome("SPY", realized_pnl=-10.0, max_consecutive_losses=2, cooldown_sec=900.0)
    assert cooling is False
    assert state.consecutive_losses["SPY"] == 1
    assert state.is_in_loss_streak_cooldown("SPY") is False
    assert can_execute(exec_plan, config, state, current_price=500.0, symbol="SPY") is True

    # Trade 2: Second consecutive loss of -$10 -> count=2 (TRIGGERS 15-MIN COOLDOWN)
    cooling = state.record_trade_outcome("SPY", realized_pnl=-10.0, max_consecutive_losses=2, cooldown_sec=900.0)
    assert cooling is True
    assert state.consecutive_losses["SPY"] == 2
    assert state.is_in_loss_streak_cooldown("SPY") is True

    # During cooling period: can_execute() blocks SPY
    assert can_execute(exec_plan, config, state, current_price=500.0, symbol="SPY") is False
    # But does NOT block other unaffected symbols (e.g. AAPL)
    assert can_execute(exec_plan, config, state, current_price=150.0, symbol="AAPL") is True

    # Simulate 15 minutes elapsed (901 seconds later)
    state.streak_cooling_until["SPY"] = time.time() - 1.0
    assert state.is_in_loss_streak_cooldown("SPY") is False
    assert can_execute(exec_plan, config, state, current_price=500.0, symbol="SPY") is True

    # Trade 3: Profitable trade (+$15.00) -> resets counter to 0
    state.record_trade_outcome("SPY", realized_pnl=15.0, max_consecutive_losses=2, cooldown_sec=900.0)
    assert state.consecutive_losses["SPY"] == 0


@pytest.mark.asyncio
async def test_fix2_handler_records_trade_outcome_on_stop_loss():
    """Fix 2: Verifies handler records negative outcome on software stop loss execution."""
    broker = MagicMock()
    broker.cancel_all_orders = AsyncMock()
    broker.close_position = AsyncMock()
    broker.get_position = AsyncMock(return_value={
        "symbol": "SPY",
        "qty": "10.0",
        "side": "long",
        "avg_entry_price": "500.0",
    })

    state = LiveBotState()
    handler = RosaRojaMarketExecutionHandler(
        broker_client=broker,
        account_equity=100000.0,
        symbol="SPY",
        max_stop_loss_usd=10.0,
        state=state,
        max_consecutive_losses=2,
        consecutive_loss_cooldown_sec=900.0,
    )

    # Current price = 489.0 -> loss of (489 - 500) * 10 = -$110.00 <= -$10.00 (triggers stop loss)
    closed = await handler.check_trailing_profit(current_price=489.0, symbol="SPY")
    assert closed is True

    # Verify outcome was recorded on state
    assert state.consecutive_losses["SPY"] == 1


# ============================================================================
# SIMULACIÓN Y ALARMA TEMPRANA: Fix 2 actúa antes que Fix 1
# ============================================================================

def test_simulation_fix2_early_warning_before_fix1_daily_loss():
    """
    Simulación: trades con -$10 de pérdida por stop-loss individual.
    Demuestra que Fix 2 (racha de 2 pérdidas = -$20) se dispara primero,
    frenando la operativa 15m antes de llegar a los 5 trades (-$50) que disparan Fix 1.
    """
    state = LiveBotState()
    cfg = PortfolioRiskConfig(max_daily_loss_usd=50.0)
    risk_mgr = PortfolioRiskManager(initial_equity=1000.0, config=cfg)

    # Trade 1: Stop-loss -$10
    state.record_trade_outcome("SPY", -10.0, max_consecutive_losses=2, cooldown_sec=900.0)
    tripped, _ = risk_mgr.update_equity(990.0)
    assert state.is_in_loss_streak_cooldown("SPY") is False
    assert tripped is False

    # Trade 2: Stop-loss -$10 -> Cumulative loss: -$20
    streak_tripped = state.record_trade_outcome("SPY", -10.0, max_consecutive_losses=2, cooldown_sec=900.0)
    tripped, _ = risk_mgr.update_equity(980.0)

    # FIX 2 DISPARA ALARMA TEMPRANA AQUÍ (tras -$20 de pérdida):
    assert streak_tripped is True
    assert state.is_in_loss_streak_cooldown("SPY") is True

    # FIX 1 AÚN NO SE DISPARA (falta perder -$30 más):
    assert tripped is False
    assert risk_mgr.circuit_breaker_tripped is False

    # Si transcurren 15 min y continúa perdiendo hasta 5 trades (-$50):
    risk_mgr.update_equity(970.0)  # Trade 3 (-$30)
    risk_mgr.update_equity(960.0)  # Trade 4 (-$40)
    tripped, reason = risk_mgr.update_equity(950.0)  # Trade 5 (-$50)

    # FIX 1 SE DISPARA FINALMENTE COMO LÍMITE DURO DE CUENTA:
    assert tripped is True
    assert "CRITICAL: Maximum Daily Loss Limit Reached!" in reason


# ============================================================================
# GARANTÍA DE NO INTERFERENCIA: EMERGENCY_FLUSH NUNCA ES BLOQUEADO
# ============================================================================

def test_emergency_flush_never_blocked_by_circuit_breaker_or_streak_cooldown():
    """Garantiza que EMERGENCY_FLUSH se ejecuta incluso con disyuntor de cuenta y cooldown activos."""
    state = LiveBotState()
    state.portfolio_circuit_breaker_tripped = True
    state.mark_closing("SPY")
    state.mark_close_confirmed("SPY")
    state.streak_cooling_until["SPY"] = time.time() + 900.0  # Streak cooling active

    config = LiveBotConfig(symbol="SPY")

    # Regular EXECUTE plan is completely blocked
    exec_plan = MagicMock(action="EXECUTE")
    assert can_execute(exec_plan, config, state, current_price=500.0, symbol="SPY") is False

    # EMERGENCY_FLUSH plan is 100% permitted through
    flush_plan = ExecutionPlan.EMERGENCY_FLUSH(reason="Critical Deviation")
    assert can_execute(flush_plan, config, state, current_price=500.0, symbol="SPY") is True
