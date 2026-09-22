"""Unit tests for Multi-Asset Portfolio Gating and Concurrent Position Management."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from iot_machine_learning.infrastructure.adapters.market.zephyr.config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.zephyr.runners import LiveBotRunner
from iot_machine_learning.infrastructure.adapters.market.zephyr.execution import (
    can_execute,
    get_current_mid,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.models import (
    LiveBotState,
)


def test_multi_asset_position_tracking():
    """LiveBotState tracks individual positions and active count across basket."""
    state = LiveBotState()
    assert state.active_positions_count == 0

    state.set_position("SPY", 7.0)
    state.set_position("QQQ", -5.0)
    state.set_position("NVDA", 0.0)

    assert state.get_position("SPY") == 7.0
    assert state.get_position("QQQ") == -5.0
    assert state.get_position("NVDA") == 0.0
    assert state.active_positions_count == 2


def test_per_symbol_position_gating():
    """can_execute prevents re-entering a symbol with an open position while allowing other assets."""
    config = LiveBotConfig(
        symbols=["SPY", "QQQ", "NVDA"],
        max_concurrent_positions=2,
        phi_moe_threshold=0.30,
    )
    state = LiveBotState()
    state.last_phi_moe = 0.50

    # Open position in SPY
    state.set_position("SPY", 5.0)

    mock_plan = MagicMock()
    mock_plan.action = "EXECUTE"

    # SPY should be blocked because position != 0
    assert not can_execute(mock_plan, config, state, current_price=580.0, symbol="SPY")

    # QQQ should be allowed because position == 0 and active_positions_count (1) < 2
    assert can_execute(mock_plan, config, state, current_price=490.0, symbol="QQQ")


def test_max_concurrent_positions_portfolio_limit():
    """can_execute strictly blocks new entries once max_concurrent_positions limit is reached."""
    config = LiveBotConfig(
        symbols=["SPY", "QQQ", "NVDA", "AAPL"],
        max_concurrent_positions=2,
        phi_moe_threshold=0.30,
    )
    state = LiveBotState()
    state.last_phi_moe = 0.50

    # Fill portfolio capacity (2 active positions)
    state.set_position("SPY", 5.0)
    state.set_position("QQQ", 3.0)
    assert state.active_positions_count == 2

    mock_plan = MagicMock()
    mock_plan.action = "EXECUTE"

    # NVDA and AAPL must both be blocked by portfolio capacity gating
    assert not can_execute(mock_plan, config, state, current_price=180.0, symbol="NVDA")
    assert not can_execute(mock_plan, config, state, current_price=240.0, symbol="AAPL")

    # If SPY closes (active count drops to 1), NVDA is now allowed
    state.set_position("SPY", 0.0)
    assert state.active_positions_count == 1
    assert can_execute(mock_plan, config, state, current_price=180.0, symbol="NVDA")


def test_multi_symbol_feed_mid_price_lookup():
    """get_current_mid correctly retrieves per-symbol quotes from multiplexed feed."""
    mock_feed = MagicMock()
    quotes = {
        "SPY": 582.45,
        "QQQ": 491.20,
        "NVDA": 182.10,
    }
    mock_feed.get_mid_price.side_effect = lambda s: quotes.get(s, None)

    assert get_current_mid(mock_feed, "SPY") == 582.45
    assert get_current_mid(mock_feed, "QQQ") == 491.20
    assert get_current_mid(mock_feed, "NVDA") == 182.10


@pytest.mark.asyncio
async def test_live_runner_multi_asset_initialization():
    """LiveBotRunner initializes isolated extractors and engines for each symbol in basket."""
    cfg = LiveBotConfig(
        symbols=["SPY", "QQQ", "NVDA", "AAPL"],
        rosa_roja_enabled=True,
    )
    mock_account = AsyncMock()
    mock_account.get_equity.return_value = 100000.0
    mock_client = AsyncMock()
    mock_feed = MagicMock()

    runner = LiveBotRunner(
        cfg,
        feed=mock_feed,
        order_client=mock_client,
        account=mock_account,
    )
    await runner.initialize()

    assert set(runner._symbol_extractors.keys()) == {"SPY", "QQQ", "NVDA", "AAPL"}
    assert set(runner._symbol_engines.keys()) == {"SPY", "QQQ", "NVDA", "AAPL"}
    assert runner._symbol_extractors["SPY"] is not runner._symbol_extractors["QQQ"]
    assert runner._symbol_engines["NVDA"] is not runner._symbol_engines["AAPL"]


def test_can_execute_with_real_trajectory_object():
    """can_execute safely extracts side from a real Trajectory with TerminalState without AttributeError."""
    import numpy as np
    from iot_machine_learning.domain.entities.rosa_roja.execution import (
        ActionEnvelope,
        ExecutionPlan,
    )
    from iot_machine_learning.domain.entities.rosa_roja.movement import (
        Movement,
        RhythmSignature,
    )
    from iot_machine_learning.domain.entities.rosa_roja.trajectory import (
        TerminalState,
        Trajectory,
    )

    config = LiveBotConfig(symbol="SPY", phi_moe_threshold=0.30)
    state = LiveBotState()
    state.last_phi_moe = 0.50

    rs = RhythmSignature(
        tempo_ratio=1.0,
        velocity_delta=0.0,
        acceleration=0.0,
        phase_angle=0.0,
        entropy_rate=0.0,
    )
    m = Movement(
        delta_state=np.array([1.5, 0.1]),
        delta_time=1.0,
        velocity=1.0,
        direction=np.array([1.0, 0.0]),
        rhythm_signature=rs,
        mahalanobis_distance=0.5,
        timestamp=100.0,
    )
    traj = Trajectory(
        movements=(m,),
        coherence_score=0.9,
        invalidation_step=None,
        terminal_state=TerminalState(
            state_vector=np.array([585.0, 0.5]),
            step_index=15,
            confidence=0.8,
        ),
    )
    envelope = ActionEnvelope(magnitude=0.5, bounds={}, max_steps=15, metadata={})
    plan = ExecutionPlan.EXECUTE(traj, confidence=0.85, envelope=envelope)

    # current_price 580.0 < terminal_price 585.0 -> side is "buy"
    allowed = can_execute(plan, config, state, current_price=580.0, symbol="SPY")
    assert allowed is True
    assert traj.side == "buy"


