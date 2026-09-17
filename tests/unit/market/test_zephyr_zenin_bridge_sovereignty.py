"""Unit tests verifying Zephyr-ZENIN Bridge Sovereignty and Zero-GC efficiency."""

import asyncio
from unittest.mock import AsyncMock, MagicMock
import numpy as np
import pytest

from iot_machine_learning.domain.entities.market.data_status import DataStatus
from iot_machine_learning.domain.entities.market.observations import Quote
from iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja.features import (
    MarketFeatureExtractor,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.execution import (
    process_observation_pipeline,
)
from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.domain.execution import (
    ActionEnvelope,
    ExecutionPlan,
)
from iot_machine_learning.infrastructure.ml.master_engine.master_equation import (
    compute_certeza,
    compute_momentum_veto,
)


def _make_quote(ts: float, bid: float = 500.0, ask: float = 500.1) -> Quote:
    return Quote(
        symbol="SPY",
        timestamp=ts,
        data_status=DataStatus.REALTIME,
        source_provider="alpaca",
        bid=bid,
        ask=ask,
        bid_size=10.0,
        ask_size=10.0,
    )


def test_market_feature_extractor_buffer_reuse():
    """Verifica que el extractor reutilice el buffer numpy evitando presiones de GC."""
    extractor = MarketFeatureExtractor()
    quote1 = _make_quote(1000.0, 500.0, 500.1)
    buf1, dt1 = extractor.process(quote1)

    assert isinstance(buf1, np.ndarray)
    assert buf1.shape == (10,)

    # El buffer interno debe ser la misma referencia
    assert buf1 is extractor._state_buffer

    quote2 = _make_quote(1001.0, 500.2, 500.3)
    buf2, dt2 = extractor.process(quote2)

    assert buf2 is extractor._state_buffer
    assert dt2 == 1.0


def test_dynamic_momentum_veto_volatility_adaptation():
    """Verifica que el umbral de momentum se adapte dinámicamente con la volatilidad del mercado."""
    # Señal moderada con volatilidad baja
    pass_low_vol = compute_momentum_veto(
        ds_dt_ema=0.01, magnitud=0.1, tau_mom=0.5, sigma_mom=0.001, sigma_market=0.0005
    )
    assert pass_low_vol == 1.0

    # Misma señal en régimen de alta volatilidad (ruido de mercado alto): se expande la zona muerta y veta
    veto_high_vol = compute_momentum_veto(
        ds_dt_ema=0.01, magnitud=0.1, tau_mom=0.5, sigma_mom=0.001, sigma_market=0.005
    )
    assert veto_high_vol == 0.0


@pytest.mark.asyncio
async def test_zephyr_blind_obedience_emergency_flush():
    """Verifica que Zephyr liquide inmediatamente ante orden de EMERGENCY_FLUSH sin reevaluar PnL."""
    runner = MagicMock()
    runner.config.symbol = "SPY"
    runner._symbol_extractors = {}
    runner._symbol_engines = {}
    runner._feature_extractor = MarketFeatureExtractor()
    runner._get_current_mid = MagicMock(return_value=500.0)
    runner._last_eval_logs = {}
    runner._state = MagicMock()
    runner._handler = MagicMock()
    runner._handler.dispatch_execution = AsyncMock(return_value=True)

    # Motor ZENIN emite EMERGENCY_FLUSH
    mock_engine = MagicMock()
    flush_plan = ExecutionPlan(
        action="EMERGENCY_FLUSH",
        chosen_trajectory=None,
        global_confidence=0.1,
        envelope=None,
        invalidation_step=None,
        regime_alert=True,
        veto_details={"reason": "Volatility_Regime_Capitulation"},
    )
    mock_engine.process_event = MagicMock(return_value=flush_plan)
    runner._engine = mock_engine

    quote = _make_quote(1000.0)
    await process_observation_pipeline(quote, runner)

    # El handler debe haber sido invocado de inmediato para liquidar
    runner._handler.dispatch_execution.assert_called_once_with(flush_plan, symbol="SPY")
    runner._state.set_position.assert_called_once_with("SPY", 0.0)


@pytest.mark.asyncio
async def test_zephyr_blind_obedience_execute():
    """Verifica que Zephyr despache la orden cuando ZENIN emite EXECUTE sin agregar heurísticas propias."""
    runner = MagicMock()
    runner.config.symbol = "SPY"
    runner.config.lot_size = 1.0
    runner._symbol_extractors = {}
    runner._symbol_engines = {}
    runner._feature_extractor = MarketFeatureExtractor()
    runner._get_current_mid = MagicMock(return_value=500.0)
    runner._last_eval_logs = {}
    runner._state = MagicMock()
    runner._state.trades_count = 0
    runner._can_execute = MagicMock(return_value=True)
    runner._handler = MagicMock()
    runner._handler.dispatch_execution = AsyncMock(return_value=True)
    runner._log_execution = AsyncMock()
    runner._save_state = AsyncMock()

    # Motor ZENIN emite EXECUTE con ActionEnvelope
    mock_engine = MagicMock()
    envelope = ActionEnvelope(
        magnitude=0.5,
        bounds={"target_pct": 0.008, "stop_pct": 0.004},
        max_steps=10,
        metadata={},
    )
    exec_plan = ExecutionPlan(
        action="EXECUTE",
        chosen_trajectory=MagicMock(),
        global_confidence=0.88,
        envelope=envelope,
        invalidation_step=None,
        regime_alert=False,
        veto_details={},
    )
    mock_engine.process_event = MagicMock(return_value=exec_plan)
    runner._engine = mock_engine

    quote = _make_quote(1000.0)
    await process_observation_pipeline(quote, runner)

    # Verificamos despacho directo al broker
    runner._handler.dispatch_execution.assert_called_once()
    call_args = runner._handler.dispatch_execution.call_args
    assert call_args[0][0].action == "EXECUTE"
    assert call_args[1]["symbol"] == "SPY"
    assert runner._log_execution.called
    assert runner._save_state.called


@pytest.mark.asyncio
async def test_zephyr_directional_action_buy_trigger():
    """Verifica que un action > 0 de la Ecuación Maestra dispare un trigger BUY."""
    runner = MagicMock()
    runner.config.symbol = "SPY"
    runner.config.lot_size = 1.0
    runner._symbol_extractors = {}
    runner._symbol_engines = {}
    runner._feature_extractor = MarketFeatureExtractor()
    runner._get_current_mid = MagicMock(return_value=500.0)
    runner._last_eval_logs = {}
    runner._state = MagicMock()
    runner._state.trades_count = 0
    runner._can_execute = MagicMock(return_value=True)
    runner._handler = MagicMock()
    runner._handler.dispatch_execution = AsyncMock(return_value=True)
    runner._log_execution = AsyncMock()
    runner._save_state = AsyncMock()

    mock_engine = MagicMock()
    # Ecuación Maestra emite action = 1.0 (Long)
    exec_plan = ExecutionPlan(
        action=1.0,
        chosen_trajectory=MagicMock(),
        global_confidence=0.85,
        envelope=None,
        invalidation_step=None,
        regime_alert=False,
        veto_details={},
    )
    mock_engine.process_event = MagicMock(return_value=exec_plan)
    runner._engine = mock_engine

    quote = _make_quote(1000.0)
    await process_observation_pipeline(quote, runner)

    runner._handler.dispatch_execution.assert_called_once()
    dispatched_plan = runner._handler.dispatch_execution.call_args[0][0]
    assert dispatched_plan.action == "EXECUTE"
    assert dispatched_plan.side == "buy"


@pytest.mark.asyncio
async def test_zephyr_directional_action_sell_trigger():
    """Verifica que un action < 0 de la Ecuación Maestra dispare un trigger SELL."""
    runner = MagicMock()
    runner.config.symbol = "SPY"
    runner.config.lot_size = 1.0
    runner._symbol_extractors = {}
    runner._symbol_engines = {}
    runner._feature_extractor = MarketFeatureExtractor()
    runner._get_current_mid = MagicMock(return_value=500.0)
    runner._last_eval_logs = {}
    runner._state = MagicMock()
    runner._state.trades_count = 0
    runner._can_execute = MagicMock(return_value=True)
    runner._handler = MagicMock()
    runner._handler.dispatch_execution = AsyncMock(return_value=True)
    runner._log_execution = AsyncMock()
    runner._save_state = AsyncMock()

    mock_engine = MagicMock()
    # Ecuación Maestra emite action = -1.0 (Short)
    exec_plan = ExecutionPlan(
        action=-1.0,
        chosen_trajectory=MagicMock(),
        global_confidence=0.90,
        envelope=None,
        invalidation_step=None,
        regime_alert=False,
        veto_details={},
    )
    mock_engine.process_event = MagicMock(return_value=exec_plan)
    runner._engine = mock_engine

    quote = _make_quote(1000.0)
    await process_observation_pipeline(quote, runner)

    runner._handler.dispatch_execution.assert_called_once()
    dispatched_plan = runner._handler.dispatch_execution.call_args[0][0]
    assert dispatched_plan.action == "EXECUTE"
    assert dispatched_plan.side == "sell"


@pytest.mark.asyncio
async def test_zephyr_directional_action_hold_trigger():
    """Verifica que un action == 0 de la Ecuación Maestra fuerce HOLD y no despache orden."""
    runner = MagicMock()
    runner.config.symbol = "SPY"
    runner._symbol_extractors = {}
    runner._symbol_engines = {}
    runner._feature_extractor = MarketFeatureExtractor()
    runner._get_current_mid = MagicMock(return_value=500.0)
    runner._last_eval_logs = {}
    runner._state = MagicMock()
    runner._handler = MagicMock()
    runner._handler.dispatch_execution = AsyncMock(return_value=True)

    mock_engine = MagicMock()
    # Ecuación Maestra emite action = 0.0 (Hold/Flat)
    exec_plan = ExecutionPlan(
        action=0.0,
        chosen_trajectory=None,
        global_confidence=0.45,
        envelope=None,
        invalidation_step=None,
        regime_alert=False,
        veto_details={},
    )
    mock_engine.process_event = MagicMock(return_value=exec_plan)
    runner._engine = mock_engine

    quote = _make_quote(1000.0)
    await process_observation_pipeline(quote, runner)

    runner._handler.dispatch_execution.assert_not_called()


def test_master_orchestrator_dependency_injection():
    """Verifica que el bot instancie MasterEquationOrchestrator y no use directamente RosaRojaEngine."""
    from iot_machine_learning.infrastructure.adapters.market.zephyr.execution import (
        create_default_master_orchestrator,
    )
    from iot_machine_learning.infrastructure.ml.master_engine.orchestrator import (
        MasterEquationOrchestrator,
    )
    from iot_machine_learning.infrastructure.adapters.market.zephyr.config import (
        LiveBotConfig,
    )

    config = LiveBotConfig(symbols=["SPY", "QQQ"])
    orchestrator = create_default_master_orchestrator(config)

    assert isinstance(orchestrator, MasterEquationOrchestrator)
    # Verifica que el sub-componente interno del orquestador sea el motor de trayectoria
    assert hasattr(orchestrator, "_rosa_roja")

