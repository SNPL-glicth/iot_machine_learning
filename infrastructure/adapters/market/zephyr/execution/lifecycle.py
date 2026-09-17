"""Lifecycle and component factory functions for LiveBotRunner."""

from __future__ import annotations

import logging
import os
import signal
from collections.abc import Callable
from typing import Any

from iot_machine_learning.infrastructure.adapters.market.alpaca.account import AlpacaAccount
from iot_machine_learning.infrastructure.adapters.market.alpaca.order_client import (
    AlpacaOrderClient,
)
from iot_machine_learning.infrastructure.adapters.market.alpaca.ws_feed import AlpacaWSFeed
from iot_machine_learning.infrastructure.adapters.market.binance.account import BinanceAccount
from iot_machine_learning.infrastructure.adapters.market.binance.order_client import (
    BinanceOrderClient,
)
from iot_machine_learning.infrastructure.adapters.market.binance.ws_feed import BinanceWSFeed
from iot_machine_learning.infrastructure.adapters.market.zephyr.config import LiveBotConfig

logger = logging.getLogger(__name__)


def create_feed(
    config: LiveBotConfig,
    on_observation: Callable | None = None,
    on_metrics: Callable | None = None,
    on_state_change: Callable | None = None,
) -> Any:
    """Crea el feed de mercado según el broker configurado."""
    if config.broker == "alpaca":
        feed_symbols = config.symbols if config.symbols else config.symbol
        return AlpacaWSFeed(
            symbol=feed_symbols,
            api_key=config.alpaca_api_key or "",
            api_secret=config.alpaca_secret_key or "",
            data_feed=config.alpaca_data_feed,
            include_trades=True,
            include_quotes=True,
            include_bars=True,
            bar_interval="1Min",
            max_queue_size=config.ws_max_queue_size,
            on_observation=on_observation,
            on_metrics=on_metrics,
            on_state_change=on_state_change,
        )
    return BinanceWSFeed(
        symbol=config.symbol,
        testnet=config.testnet,
        depth_speed=config.depth_speed,
        include_trades=config.include_trades,
        include_book_ticker=config.include_book_ticker,
        include_kline=config.include_kline,
        kline_interval=config.kline_interval,
        max_queue_size=config.ws_max_queue_size,
        snapshot_interval_sec=config.ob_snapshot_interval_sec,
        on_observation=on_observation,
        on_metrics=on_metrics,
        on_state_change=on_state_change,
    )


def get_binance_api_key() -> str:
    key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_TESTNET_API_KEY")
    if not key:
        raise ValueError("BINANCE_API_KEY or BINANCE_TESTNET_API_KEY not set")
    return key


def get_binance_api_secret() -> str:
    secret = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_TESTNET_API_SECRET")
    if not secret:
        raise ValueError("BINANCE_API_SECRET or BINANCE_TESTNET_API_SECRET not set")
    return secret


def create_order_client(config: LiveBotConfig) -> Any:
    """Crea el cliente de órdenes según el broker configurado."""
    if config.broker == "alpaca":
        return AlpacaOrderClient(
            api_key=config.alpaca_api_key or "",
            api_secret=config.alpaca_secret_key or "",
            base_url=config.alpaca_api_base_url,
            data_feed=config.alpaca_data_feed,
        )
    return BinanceOrderClient(
        api_key=get_binance_api_key(),
        api_secret=get_binance_api_secret(),
        testnet=config.testnet,
    )


def create_account(config: LiveBotConfig, order_client: Any) -> Any:
    """Crea la cuenta según el broker configurado."""
    client = order_client or create_order_client(config)
    if config.broker == "alpaca":
        return AlpacaAccount(client=client, auto_sync_interval=5.0)
    return BinanceAccount(client=client, symbol=config.symbol, auto_sync_interval=5.0)


def _create_internal_trajectory_engine(config: LiveBotConfig) -> Any:
    """Internal trajectory and epistemic jury engine construction."""
    from iot_machine_learning.infrastructure.ml.adapters import (
        BaseExpertAdapter, KalmanExpertAdapter, StatisticalExpertAdapter, TaylorExpertAdapter,
    )
    from iot_machine_learning.infrastructure.ml.engines.kalman.engine import KalmanPredictionEngine
    from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.domain.trajectory_tracker import TrajectoryTracker
    from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.engine import RosaRojaEngine
    from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.module1_ingestion import MahalanobisFilter
    from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.module3_moe_gating import MultiplicativeMoEGating
    from iot_machine_learning.infrastructure.ml.engines.rosa_roja.algorithms.modules.rhythm_generator import RhythmTrajectoryGenerator
    from iot_machine_learning.infrastructure.ml.engines.seasonal.engine import SeasonalPredictorEngine
    from iot_machine_learning.infrastructure.ml.engines.statistical import StatisticalPredictionEngine
    from iot_machine_learning.infrastructure.ml.engines.taylor.engine import TaylorPredictionEngine

    ingestion = MahalanobisFilter(noise_threshold=3.0, history_window=100, min_samples_for_cov=20)
    rhythm = RhythmTrajectoryGenerator(
        min_trajectory_len=11, max_trajectory_len=15, top_k=4, oversample_factor=2, max_random_walk_steps=40
    )
    gating = MultiplicativeMoEGating(variance_penalty=0.5)

    jury = [
        TaylorExpertAdapter(engine=TaylorPredictionEngine()),
        KalmanExpertAdapter(engine=KalmanPredictionEngine()),
        StatisticalExpertAdapter(engine=StatisticalPredictionEngine()),
        BaseExpertAdapter(engine=SeasonalPredictorEngine(), name="seasonal", weight=0.8),
    ]
    engine = RosaRojaEngine(
        ingestion_filter=ingestion, rhythm_generator=rhythm, moe_gating=gating,
        expert_jury=jury, drift_sensors=[], outlier_reset_threshold=3, exploration_boost_events=5,
    )
    engine.gamma_exec = float(config.phi_moe_threshold)
    engine.geometric_threshold = float(config.geometric_threshold)
    max_vel_err = getattr(config, "tracker_max_vel_rel_err", None)
    engine._tracker = TrajectoryTracker(max_direction_dev_deg=120.0, max_velocity_rel_err=max_vel_err)
    return engine


def create_default_master_orchestrator(config: LiveBotConfig) -> Any:
    """Crea el orquestador maestro unificando Rosa Roja, Riesgo Estocástico y Sincronía Temporal."""
    from iot_machine_learning.infrastructure.ml.adapters import RiskEngineAdapter, TemporalEngineAdapter
    from iot_machine_learning.infrastructure.ml.master_engine import MasterEquationOrchestrator

    return MasterEquationOrchestrator(
        rosa_roja_engine=_create_internal_trajectory_engine(config),
        risk_adapter=RiskEngineAdapter(),
        temporal_adapter=TemporalEngineAdapter(),
        shadow_mode=getattr(config, "master_shadow_mode", False),
        tau_mom=getattr(config, "tau_mom", 0.5),
        sigma_mom=getattr(config, "sigma_mom", 0.001),
    )


def create_default_rosa_roja_engine(config: LiveBotConfig) -> Any:
    """Backward compatibility alias for MasterEquationOrchestrator."""
    return create_default_master_orchestrator(config)


def install_signal_handlers(shutdown_event: Any, already_installed: bool = False) -> bool:
    """Instala signal handlers para graceful shutdown."""
    if already_installed:
        return True
    def handler(signum, frame):
        logger.info("Signal received, initiating shutdown", extra={"signal": signum})
        shutdown_event.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)
    return True


