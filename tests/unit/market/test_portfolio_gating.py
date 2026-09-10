"""Unit tests for Multi-Asset Portfolio Gating and Concurrent Position Management."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.live_runner import LiveBotRunner
from iot_machine_learning.infrastructure.adapters.market.live_runner_execution import (
    can_execute,
    get_current_mid,
)
from iot_machine_learning.infrastructure.adapters.market.live_runner_models import (
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

