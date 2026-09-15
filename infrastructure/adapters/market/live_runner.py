"""LiveBotRunner -- Main event-driven runner for live algorithmic trading."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.live_runner_execution import (
    can_execute,
    get_current_mid,
    load_state,
    log_execution,
    perform_shutdown,
    save_state,
)
from iot_machine_learning.infrastructure.adapters.market.live_runner_lifecycle import (
    create_account,
    create_default_rosa_roja_engine,
    create_feed,
    create_order_client,
    install_signal_handlers,
)
from iot_machine_learning.infrastructure.adapters.market.live_runner_models import (
    ExecutionContext,
    LiveBotState,
)
from iot_machine_learning.infrastructure.adapters.market.live_runner_telemetry import (
    build_telemetry_state,
    format_status_line,
    perform_health_check,
)
from iot_machine_learning.infrastructure.adapters.market.portfolio_risk_manager import (
    PortfolioRiskConfig,
    PortfolioRiskManager,
)
from iot_machine_learning.infrastructure.adapters.market.rosa_roja_features import (
    MarketFeatureExtractor,
)
from iot_machine_learning.infrastructure.adapters.market.rosa_roja_market_handler import (
    RosaRojaMarketExecutionHandler,
)
from iot_machine_learning.infrastructure.adapters.market.telemetry_server import (
    create_telemetry_server,
)
from iot_machine_learning.infrastructure.adapters.market.trailing_profit_manager import (
    TrailingProfitConfig,
)

logger = logging.getLogger(__name__)


class LiveBotRunner:
    """Orquestador de trading live event-driven modularizado."""

    def __init__(
        self, config: LiveBotConfig, *, engine: Any | None = None, feed: Any | None = None,
        order_client: Any | None = None, account: Any | None = None,
        feature_extractor: MarketFeatureExtractor | None = None, state_path: Path | None = None,
        on_status: Callable[[str], None] | None = None,
    ):
        self.config, self._engine, self._feed = config, engine, feed
        self._order_client, self._account = order_client, account
        self._feature_extractor = feature_extractor or MarketFeatureExtractor()
        self._symbols: list[str] = list(config.symbols) if config.symbols else [config.symbol]
        self._symbol_extractors: dict[str, MarketFeatureExtractor] = {}
        self._symbol_engines: dict[str, Any] = {}
        self._state_path: Path | None = Path(state_path) if state_path else (Path(config.state_snapshot_path) if config.state_snapshot_path else None)
        self._on_status = on_status or (lambda line: logger.info(line))
        self._handler: RosaRojaMarketExecutionHandler | None = None
        self._portfolio_risk_mgr: PortfolioRiskManager | None = None
        self._state, self._running, self._shutdown_event = LiveBotState(), False, asyncio.Event()
        self._last_health_check, self._last_broadcast, self._start_time = 0.0, 0.0, time.time()
        self._last_eval_log: float = 0.0
        self._last_eval_logs: dict[str, float] = {}
        self._last_clock_check: float = 0.0
        self._latency_samples: deque[float] = deque(maxlen=1000)
        self._execution_history: list[ExecutionContext] = []
        self._audit_log_path: Path | None = None
        self._telemetry: Any | None = None
        self._signal_handlers_installed: bool = False

    async def initialize(self) -> None:
        """Inicializa los componentes requeridos."""
        self._symbols = list(self.config.symbols) if self.config.symbols else [self.config.symbol]
        self._symbol_extractors = {s: MarketFeatureExtractor(window=200) for s in self._symbols}
        self._feature_extractor = self._symbol_extractors.get(self.config.symbol, MarketFeatureExtractor(window=200))
        if self._engine is None and self.config.rosa_roja_enabled:
            self._symbol_engines = {s: self._create_rosa_roja_engine() for s in self._symbols}
            self._engine = self._symbol_engines.get(self.config.symbol)
        elif self._engine is not None:
            self._symbol_engines = {s: (self._engine if s == self.config.symbol else self._create_rosa_roja_engine()) for s in self._symbols}
        elif self._engine is None:
            raise ValueError("Engine required but rosa_roja_enabled=False")
        if self._feed is None:
            self._feed = self._create_feed()
        if self._order_client is None:
            self._order_client = self._create_order_client()
        if self._account is None:
            self._account = self._create_account()

        equity = await self._get_equity()
        trailing_cfg = (
            TrailingProfitConfig(
                activation_pnl_usd=getattr(self.config, "trailing_activation_pnl", 4.50),
                giveback_ratio=getattr(self.config, "trailing_giveback_ratio", 0.30),
                min_giveback_usd=getattr(self.config, "trailing_min_giveback", 1.80),
                max_giveback_usd=getattr(self.config, "trailing_max_giveback", 3.50),
            )
            if getattr(self.config, "use_trailing_profit", True)
            else None
        )
        self._handler = RosaRojaMarketExecutionHandler(
            broker_client=self._order_client, account_equity=equity, symbol=self.config.symbol,
            lot_size=self.config.lot_size, min_qty=self.config.min_lot_size, max_position_pct=self.config.max_position_pct,
            trailing_config=trailing_cfg,
            max_stop_loss_usd=getattr(self.config, "max_trade_loss_usd", 10.0),
            state=self._state,
            max_consecutive_losses=getattr(self.config, "max_consecutive_losses", 2),
            consecutive_loss_cooldown_sec=getattr(self.config, "consecutive_loss_cooldown_sec", 900.0),
        )
        self._handler.get_reference_price_callback = self._get_current_mid
        risk_cfg = PortfolioRiskConfig(
            profit_lock_trigger_usd=getattr(self.config, "portfolio_profit_lock_trigger", 10.0),
            max_giveback_pct=getattr(self.config, "portfolio_max_giveback_pct", 0.25),
            max_cluster_positions=getattr(self.config, "max_cluster_correlated_positions", 1),
            enable_macro_velocity_filter=getattr(self.config, "enforce_macro_velocity_alignment", True),
            max_daily_loss_usd=getattr(self.config, "max_daily_loss_usd", 50.0),
        )
        self._portfolio_risk_mgr = PortfolioRiskManager(initial_equity=equity, config=risk_cfg)
        if self.config.enable_audit_log and self.config.audit_log_path:
            self._audit_log_path = Path(self.config.audit_log_path)
            self._audit_log_path.mkdir(parents=True, exist_ok=True)
        await self._load_state()
        self._install_signal_handlers()
        if self.config.enable_metrics_export:
            self._telemetry = await create_telemetry_server(self)

    async def run(self) -> None:
        """Loop principal de consumo de ticks y toma de decisiones."""
        if not self._feed or not self._engine:
            raise RuntimeError("Runner not initialized. Call initialize() first.")
        self._running = True
        await self._check_market_session()
        await self._feed.connect()
        try:
            async for obs in self._feed.iter_observations():
                if not self._running or self._shutdown_event.is_set():
                    break
                await self._maybe_health_check()
                await self._process_observation(obs)
                await self._broadcast_telemetry()
        except asyncio.CancelledError:
            logger.info("Runner cancelled")
        except Exception as e:
            logger.error("Runner error", extra={"error": str(e)})
            self._state.last_error = str(e)
            raise
        finally:
            await self.shutdown()

    async def _process_observation(self, obs: Any) -> None:
        """Procesa una observación entrante en el pipeline de Rosa Roja."""
        sym = getattr(obs, "symbol", self.config.symbol)
        extractor = self._symbol_extractors.get(sym, self._feature_extractor)
        engine = self._symbol_engines.get(sym, self._engine)

        delta_state, delta_time = extractor.process(obs)
        if engine is None:
            return

        mid = self._get_current_mid(sym)
        if self._handler and mid > 0:
            closed = await self._handler.check_trailing_profit(mid, symbol=sym)
            if closed:
                self._state.set_position(sym, 0.0)
                if sym == self.config.symbol:
                    self._state.current_position = 0.0
                if self._account and hasattr(self._account, "sync_now"):
                    asyncio.create_task(self._account.sync_now())

        plan = engine.process_event(delta_state, delta_time)
        self._state.last_phi_moe = plan.global_confidence
        self._state.last_phi_ritmo = plan.chosen_trajectory.coherence_score if plan.chosen_trajectory else 0.0

        scores: dict[str, Any] = {}
        shadow_str = ""
        dt: dict[str, Any] = {}
        if plan.envelope and plan.envelope.metadata and "decision_trace" in plan.envelope.metadata:
            dt = plan.envelope.metadata["decision_trace"]
        elif getattr(plan, "veto_details", None) and isinstance(plan.veto_details, dict) and "decision_trace" in plan.veto_details:
            dt = plan.veto_details["decision_trace"]

        if dt:
            self._state.last_lambda_t = float(dt.get("lambda_t", 0.0))
            self._state.last_phi_ritmo = float(dt.get("phi_ritmo", self._state.last_phi_ritmo))
            scores = dt.get("expert_confidences", {})
            self._state.last_expert_votes = [
                {"name": n, "vote": round(float(s) * 2.0 - 1.0, 2), "weight": 1.0, "confidence": round(float(s), 2)}
                for n, s in scores.items()
            ]
            r_sh = dt.get("risk_engine_shadow", {})
            t_sh = dt.get("temporal_engine_shadow", {})
            if r_sh or t_sh:
                r_veto = r_sh.get("veto_riesgo", 1)
                cvar = r_sh.get("cvar_t", 0.0)
                crono = t_sh.get("lambda_crono", 0.0)
                shadow_str = f"[shadow: risk_veto={r_veto} cvar={cvar:.4f} crono={crono:.3f}]"

        now = time.time()
        last_eval = self._last_eval_logs.get(sym, 0.0)
        if now - last_eval >= 5.0 or plan.action == "EXECUTE":
            self._last_eval_logs[sym] = now
            self._last_eval_log = now
            reason = plan.veto_details.get("reason", "") if getattr(plan, "veto_details", None) else ""
            sc_str = " ".join(f"{k.split('_')[0]}:{float(v):.2f}" for k, v in scores.items()) if scores else ""
            logger.info(
                "Evaluation [%s] action=%s reason='%s' mid=%.2f phi_moe=%.3f trades=%d %s %s",
                sym, plan.action, reason, mid,
                self._state.last_phi_moe, self._state.trades_count, f"[{sc_str}]" if sc_str else "",
                shadow_str,
            )

        if (getattr(self.config, "enforce_portfolio_profit_lock", True) or getattr(self.config, "max_daily_loss_usd", 0.0) > 0) and self._portfolio_risk_mgr:
            current_eq = await self._get_equity()
            tripped, reason = self._portfolio_risk_mgr.update_equity(current_eq)
            if tripped and not self._state.portfolio_circuit_breaker_tripped:
                self._state.portfolio_circuit_breaker_tripped = True
                logger.critical("PORTFOLIO CIRCUIT BREAKER TRIPPED: %s", reason)
                if self._handler:
                    for s in self._symbols:
                        await self._handler.trigger_emergency_flush(reason, symbol=s)

        macro_vel = 0.0
        if extractor and hasattr(extractor, "state") and len(extractor.state.mid_prices) >= 5:
            p = list(extractor.state.mid_prices)
            macro_vel = (p[-1] - p[-5]) / p[-5] if p[-5] > 0 else 0.0

        if not self._can_execute(plan, symbol=sym, macro_velocity=macro_vel):
            return

        if self._handler and await self._handler.dispatch_execution(plan, symbol=sym) and plan.action == "EXECUTE":
            self._state.last_execution_time = time.time()
            self._state.last_execution_price = mid
            self._state.trades_count += 1
            await self._log_execution(plan)
            await self._save_state()
        await self._maybe_health_check()

    def _can_execute(self, plan: Any, symbol: str | None = None, macro_velocity: float = 0.0) -> bool:
        return can_execute(
            plan, self.config, self._state, self._get_current_mid(symbol),
            symbol=symbol, risk_mgr=self._portfolio_risk_mgr, macro_velocity=macro_velocity,
        )
    def _get_current_mid(self, symbol: str | None = None) -> float: return get_current_mid(self._feed, symbol=symbol or self.config.symbol)
    async def _log_execution(self, plan: Any) -> None: await log_execution(plan, self._state, self.config, self._audit_log_path, self._execution_history)
    async def _save_state(self) -> None: await save_state(self._state, self._state_path)
    async def _load_state(self) -> None: await load_state(self._state, self._state_path)
    async def shutdown(self) -> None:
        self._running = False
        self._shutdown_event.set()
        await perform_shutdown(self._handler, self._order_client, self._feed, self._account, self._state, self.config, self._state_path)
    def _create_feed(self) -> Any: return create_feed(self.config, self._on_observation_callback, self._on_feed_metrics, self._on_feed_state_change)
    def _create_order_client(self) -> Any: return create_order_client(self.config)
    def _create_account(self) -> Any: return create_account(self.config, self._order_client)
    def _create_rosa_roja_engine(self) -> Any: return create_default_rosa_roja_engine(self.config)
    def _install_signal_handlers(self) -> None: self._signal_handlers_installed = install_signal_handlers(self._shutdown_event, self._signal_handlers_installed)
    async def _get_equity(self) -> float: return float(await self._account.get_equity()) if self._account else 10000.0
    def _on_observation_callback(self, obs: Any) -> None: pass
    def _on_feed_metrics(self, metrics: dict) -> None: pass
    def _on_feed_state_change(self, old: Any, new: Any) -> None:
        logger.info("Feed state change", extra={"from": getattr(old, "value", old), "to": getattr(new, "value", new), "symbol": self.config.symbol})
    async def _build_telemetry_state(self) -> dict[str, Any]:
        return await build_telemetry_state(self._state, self.config, self._feed, self._order_client, self._account, self._latency_samples)
    async def _broadcast_telemetry(self) -> None:
        if self._telemetry and (time.time() - self._last_broadcast >= 0.1):
            self._last_broadcast = time.time()
            try:
                await self._telemetry.broadcast_state(await self._build_telemetry_state())
            except Exception as e:
                logger.debug(f"Telemetry broadcast error: {e}")
    async def _maybe_health_check(self) -> None:
        if self._account and hasattr(self._account, "is_tradeable"):
            try:
                tradeable = await self._account.is_tradeable()
                self._state.account_blocked = not tradeable
                if not tradeable:
                    logger.warning("Account health check: account is NOT tradeable (blocked or suspicious). Entry orders vetoed.")
            except Exception as e:
                logger.warning("Failed to check account tradeable status: %s", e)
                self._state.account_blocked = True

        if self._account and hasattr(self._account, "get_position"):
            try:
                symbols = getattr(self, "_symbols", [self.config.symbol])
                for sym in symbols:
                    pos = await self._account.get_position(sym)
                    self._state.set_position(sym, pos)
                if hasattr(self._account, "get_unrealized_pl"):
                    self._state.total_pnl = await self._account.get_unrealized_pl(self.config.symbol)
            except Exception as e:
                logger.debug(f"Position sync error: {e}")

        if self._portfolio_risk_mgr and (getattr(self.config, "enforce_portfolio_profit_lock", True) or getattr(self.config, "max_daily_loss_usd", 0.0) > 0):
            try:
                current_eq = await self._get_equity()
                tripped, reason = self._portfolio_risk_mgr.update_equity(current_eq)
                if tripped and not self._state.portfolio_circuit_breaker_tripped:
                    self._state.portfolio_circuit_breaker_tripped = True
                    logger.critical("PORTFOLIO CIRCUIT BREAKER TRIPPED: %s", reason)
                    if self._handler:
                        for s in self._symbols:
                            await self._handler.trigger_emergency_flush(reason, symbol=s)
            except Exception as e:
                logger.debug("Portfolio risk evaluation error: %s", e)
        await self._check_market_session()
        if time.time() - self._last_health_check >= self.config.health_check_interval_sec:
            self._last_health_check = time.time()
            await self._health_check()

    async def _check_market_session(self) -> None:
        """Monitorea el reloj de mercado de Alpaca (cierre RTH 16:00 ET)."""
        if not self._order_client or not hasattr(self._order_client, "get_clock"):
            return
        now = time.time()
        if now - self._last_clock_check < 30.0:
            return
        self._last_clock_check = now
        try:
            clock = await self._order_client.get_clock()
            is_open = bool(clock.get("is_open", False))
            next_close_str = clock.get("next_close")
            if not is_open:
                if getattr(self.config, "enforce_market_hours", False):
                    if not self._state.market_closed:
                        self._state.market_closed = True
                        self._state.market_closing_soon = True
                        logger.info("Market session is CLOSED (clock.is_open=False). Inhibiting entries.")
                        if self._handler:
                            symbols = getattr(self, "_symbols", [self.config.symbol])
                            for sym in symbols:
                                await self._handler.trigger_emergency_flush("Market Close", symbol=sym)
                else:
                    logger.debug("Market clock is_open=False, continuing in extended/paper mode.")
                return

            if self._state.market_closed or self._state.market_closing_soon:
                self._state.market_closed = False
                self._state.market_closing_soon = False
                logger.info("Market session is OPEN (clock.is_open=True). Resuming entry orders.")
                if self._portfolio_risk_mgr:
                    try:
                        self._portfolio_risk_mgr.reset_day(await self._get_equity())
                    except Exception as e:
                        logger.warning("Failed to reset daily risk baseline: %s", e)

            if next_close_str and getattr(self.config, "enforce_market_hours", False):
                from datetime import datetime
                close_dt = datetime.fromisoformat(next_close_str)
                now_dt = datetime.fromisoformat(clock.get("timestamp", datetime.now().isoformat()))
                sec_to_close = (close_dt - now_dt).total_seconds()
                if sec_to_close <= 300 and not self._state.market_closing_soon:
                    self._state.market_closing_soon = True
                    logger.warning(
                        "Market closing in %.1f minutes (~15:55 ET). Inhibiting new entry orders.",
                        sec_to_close / 60.0,
                    )
                elif sec_to_close <= 10 and not self._state.market_closed:
                    self._state.market_closed = True
                    self._state.market_closing_soon = True
                    logger.warning("Market session closing imminently (16:00 ET). Flattening positions.")
                    if self._handler:
                        symbols = getattr(self, "_symbols", [self.config.symbol])
                        for sym in symbols:
                            await self._handler.trigger_emergency_flush("End of Day Close", symbol=sym)
        except Exception as e:
            logger.debug(f"Error checking market session: {e}")
    async def _health_check(self) -> None:
        await perform_health_check(self._state, self.config, self._feed, self._latency_samples, self._running, getattr(self, "on_health_check", None))
    def get_status_line(self) -> str: return format_status_line(self._state, self._feed, self._start_time)


async def create_live_bot(config: LiveBotConfig | None = None, **kwargs) -> LiveBotRunner:
    """Factory para crear e inicializar LiveBotRunner."""
    runner = LiveBotRunner(config or LiveBotConfig(**kwargs))
    await runner.initialize()
    return runner


async def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="ZENIN Live Bot - Event-driven trading")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--testnet", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--config", type=Path)
    a = p.parse_args()
    cfg = LiveBotConfig.from_file(a.config) if a.config else LiveBotConfig(symbol=a.symbol, testnet=a.testnet, dry_run=a.dry_run)
    runner = await create_live_bot(cfg)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
