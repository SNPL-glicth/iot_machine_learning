"""LiveBotRunner -- Main event-driven runner for live algorithmic trading."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from iot_machine_learning.infrastructure.adapters.market.zephyr.config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.zephyr.execution import (
    can_execute, check_market_session, create_account, create_default_master_orchestrator,
    create_feed, create_order_client, get_current_mid, install_signal_handlers,
    load_state, log_execution, perform_shutdown, process_observation_pipeline,
    save_state, sync_and_check_health,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.models import (
    ContextualMemoryManager, ExecutionContext, LiveBotState,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja_execution.features import MarketFeatureExtractor
from iot_machine_learning.infrastructure.adapters.market.zephyr.engines.rosa_roja_execution.market_handler import RosaRojaMarketExecutionHandler
from iot_machine_learning.infrastructure.adapters.market.zephyr.risk.portfolio_risk_manager import PortfolioRiskConfig, PortfolioRiskManager
from iot_machine_learning.infrastructure.adapters.market.zephyr.risk.trailing_profit_manager import TrailingProfitConfig
from iot_machine_learning.infrastructure.adapters.market.zephyr.telemetry.builder import build_telemetry_state, format_status_line
from iot_machine_learning.infrastructure.adapters.market.zephyr.telemetry.server import TelemetryBroadcaster, create_telemetry_server
from iot_machine_learning.infrastructure.adapters.persistence.weaviate_telemetry import WeaviateTelemetryStore

logger = logging.getLogger(__name__)


class LiveBotRunner:
    """Orquestador de trading live event-driven modularizado."""

    def __init__(
        self, config: LiveBotConfig, *, engine: Any | None = None, feed: Any | None = None,
        order_client: Any | None = None, account: Any | None = None,
        feature_extractor: MarketFeatureExtractor | None = None, state_path: Path | None = None,
        on_status: Callable[[str], None] | None = None,
    ):
        self.config, self._engine, self._feed, self._order_client, self._account = config, engine, feed, order_client, account
        self._feature_extractor = feature_extractor or MarketFeatureExtractor()
        self._symbols = list(config.symbols) if config.symbols else [config.symbol]
        self._symbol_extractors, self._symbol_engines = {}, {}
        self._state_path = Path(state_path) if state_path else (Path(config.state_snapshot_path) if config.state_snapshot_path else None)
        self._on_status = on_status or (lambda line: logger.info(line))
        self._handler: RosaRojaMarketExecutionHandler | None = None
        self._portfolio_risk_mgr: PortfolioRiskManager | None = None
        self._state, self._running, self._shutdown_event = LiveBotState(), False, asyncio.Event()
        self._last_health_check, self._last_broadcast, self._start_time = 0.0, 0.0, time.time()

        self._last_eval_logs: dict[str, float] = {}
        self._latency_samples: deque[float] = deque(maxlen=1000)
        self._execution_history: deque[ExecutionContext] = deque(maxlen=1000)
        self._market_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._audit_log_path: Path | None = None
        self._telemetry: TelemetryBroadcaster | None = None
        self._signal_handlers_installed = False
        self._memory_mgr = ContextualMemoryManager(max_daily_loss=getattr(config, "max_daily_loss_usd", 50.0))
        self._state.on_trade_outcome = self._memory_mgr.update_from_execution
        self._weaviate_store = WeaviateTelemetryStore(url=getattr(config, "weaviate_url", "http://localhost:8080"))

    async def initialize(self) -> None:
        """Inicializa los componentes requeridos."""
        self._symbols = list(self.config.symbols) if self.config.symbols else [self.config.symbol]
        self._symbol_extractors = {s: MarketFeatureExtractor(window=200) for s in self._symbols}
        self._feature_extractor = self._symbol_extractors.get(self.config.symbol, MarketFeatureExtractor(window=200))
        if self._engine is None:
            self._symbol_engines = {s: self._create_master_orchestrator() for s in self._symbols}
            self._engine = self._symbol_engines.get(self.config.symbol)
        else:
            self._symbol_engines = {s: (self._engine if s == self.config.symbol else self._create_master_orchestrator()) for s in self._symbols}

        if self._feed is None: self._feed = create_feed(self.config, lambda o: None, lambda m: None, lambda a, b: None)
        if self._order_client is None: self._order_client = create_order_client(self.config)
        if self._account is None: self._account = create_account(self.config, self._order_client)
        equity = await self._get_equity()
        trailing_cfg = TrailingProfitConfig(
            activation_pnl_usd=getattr(self.config, "trailing_activation_pnl", 4.50), giveback_ratio=getattr(self.config, "trailing_giveback_ratio", 0.30),
            min_giveback_usd=getattr(self.config, "trailing_min_giveback", 1.80), max_giveback_usd=getattr(self.config, "trailing_max_giveback", 3.50),
        ) if getattr(self.config, "use_trailing_profit", True) else None

        self._handler = RosaRojaMarketExecutionHandler(
            broker_client=self._order_client, account_equity=equity, symbol=self.config.symbol,
            lot_size=self.config.lot_size, min_qty=self.config.min_lot_size, max_qty=self.config.max_lot_size, max_position_pct=self.config.max_position_pct,
            trailing_config=trailing_cfg, max_stop_loss_usd=getattr(self.config, "max_trade_loss_usd", 10.0), state=self._state,
            max_consecutive_losses=getattr(self.config, "max_consecutive_losses", 2), consecutive_loss_cooldown_sec=getattr(self.config, "consecutive_loss_cooldown_sec", 900.0),
        )
        self._handler.get_reference_price_callback = self._get_current_mid
        risk_cfg = PortfolioRiskConfig(
            profit_lock_trigger_usd=getattr(self.config, "portfolio_profit_lock_trigger", 10.0), max_giveback_pct=getattr(self.config, "portfolio_max_giveback_pct", 0.25),
            max_cluster_positions=getattr(self.config, "max_cluster_correlated_positions", 1), enable_macro_velocity_filter=getattr(self.config, "enforce_macro_velocity_alignment", True),
            max_daily_loss_usd=getattr(self.config, "max_daily_loss_usd", 50.0),
        )
        self._portfolio_risk_mgr = PortfolioRiskManager(initial_equity=equity, config=risk_cfg, clusters=getattr(self.config, "asset_clusters", None))
        if self.config.enable_audit_log and self.config.audit_log_path:
            self._audit_log_path = Path(self.config.audit_log_path)
            self._audit_log_path.mkdir(parents=True, exist_ok=True)
        await save_state(self._state, self._state_path) if not self._state_path or not self._state_path.exists() else await load_state(self._state, self._state_path)
        self._signal_handlers_installed = install_signal_handlers(self._shutdown_event, self._signal_handlers_installed)
        if self.config.enable_metrics_export: self._telemetry = await create_telemetry_server(self)
        await self._weaviate_store.start()

    async def run(self) -> None:
        """Loop principal desacoplado en tareas concurrentes de ingesta, proceso y telemetría."""
        if not self._feed or not self._engine: raise RuntimeError("Runner not initialized. Call initialize() first.")
        self._running = True
        await self._check_market_session()
        await self._feed.connect()
        try:
            await asyncio.gather(self._ingest_worker(), self._processing_worker(), self._telemetry_worker())
        except asyncio.CancelledError: logger.info("Runner tasks cancelled")
        except Exception as e:
            logger.error("Runner error", extra={"error": str(e)})
            self._state.last_error = str(e)
            raise
        finally: await self.shutdown()

    async def _ingest_worker(self) -> None:
        assert self._feed is not None, "Feed must be initialized before _ingest_worker"
        async for obs in self._feed.iter_observations():
            if not self._running or self._shutdown_event.is_set(): break
            if self._market_queue.full():
                try: self._market_queue.get_nowait()
                except asyncio.QueueEmpty: pass
            await self._market_queue.put(obs)

    async def _processing_worker(self) -> None:
        while self._running and not self._shutdown_event.is_set():
            try:
                obs = await asyncio.wait_for(self._market_queue.get(), timeout=1.0)
                await self._process_observation(obs)
                self._market_queue.task_done()
            except asyncio.TimeoutError: continue

    async def _telemetry_worker(self) -> None:
        while self._running and not self._shutdown_event.is_set():
            await self._maybe_health_check()
            await self._broadcast_telemetry()
            await asyncio.sleep(0.1)

    async def _process_observation(self, obs: Any) -> None: await process_observation_pipeline(obs, self)
    def _can_execute(self, plan: Any, symbol: str | None = None) -> bool:
        return can_execute(plan, self.config, self._state, self._get_current_mid(symbol), symbol=symbol)
    def _get_current_mid(self, symbol: str | None = None) -> float: return get_current_mid(self._feed, symbol=symbol or self.config.symbol)
    async def _log_execution(self, plan: Any) -> None: await log_execution(plan, self._state, self.config, self._audit_log_path, self._execution_history)
    async def _save_state(self) -> None: await save_state(self._state, self._state_path)
    async def _load_state(self) -> None: await load_state(self._state, self._state_path)
    def _create_master_orchestrator(self) -> Any: return create_default_master_orchestrator(self.config)

    async def shutdown(self) -> None:
        self._running = False
        self._shutdown_event.set()
        await perform_shutdown(self._handler, self._order_client, self._feed, self._account, self._state, self.config, self._state_path, self._weaviate_store)

    async def _get_equity(self) -> float:
        eq = float(await self._account.get_equity()) if self._account else 10000.0
        self._state.equity = eq
        if getattr(self, "_handler", None): self._handler.update_equity(eq)
        return eq

    async def _broadcast_telemetry(self) -> None:
        if self._telemetry and (time.time() - self._last_broadcast >= 0.1):
            self._last_broadcast = time.time()
            try: await self._telemetry.broadcast_state(await build_telemetry_state(self._state, self.config, self._feed, self._order_client, self._account, self._latency_samples))
            except Exception as e: logger.debug("Telemetry broadcast error: %s", e)

    async def _maybe_health_check(self) -> None:
        self._last_health_check = await sync_and_check_health(
            self._state, self.config, self._account, self._handler, self._portfolio_risk_mgr,
            self._symbols, self._get_equity, self._feed, self._latency_samples, self._running, self._last_health_check,
        )

    async def _check_market_session(self) -> None:
        await check_market_session(self._state, self.config, self._order_client, self._handler, self._portfolio_risk_mgr, self._symbols, self._get_equity)
    def get_status_line(self) -> str: return format_status_line(self._state, self._feed, self._start_time)

async def create_live_bot(config: LiveBotConfig | None = None, **kwargs) -> LiveBotRunner:
    runner = LiveBotRunner(config or LiveBotConfig(**kwargs))
    await runner.initialize()
    return runner

