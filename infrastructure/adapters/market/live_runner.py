"""LiveBotRunner -- Main event-driven runner for live algorithmic trading."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.live_runner_models import ExecutionContext, LiveBotState
from iot_machine_learning.infrastructure.adapters.market.live_runner_lifecycle import (
    create_account, create_default_rosa_roja_engine, create_feed, create_order_client, install_signal_handlers,
)
from iot_machine_learning.infrastructure.adapters.market.live_runner_execution import (
    can_execute, get_current_mid, load_state, log_execution, perform_shutdown, save_state,
)
from iot_machine_learning.infrastructure.adapters.market.live_runner_telemetry import (
    build_telemetry_state, format_status_line, perform_health_check,
)
from iot_machine_learning.infrastructure.adapters.market.rosa_roja_features import MarketFeatureExtractor
from iot_machine_learning.infrastructure.adapters.market.rosa_roja_market_handler import RosaRojaMarketExecutionHandler
from iot_machine_learning.infrastructure.adapters.market.telemetry_server import create_telemetry_server

logger = logging.getLogger(__name__)


class LiveBotRunner:
    """Orquestador de trading live event-driven modularizado."""

    def __init__(
        self, config: LiveBotConfig, *, engine: Optional[Any] = None, feed: Optional[Any] = None,
        order_client: Optional[Any] = None, account: Optional[Any] = None,
        feature_extractor: Optional[MarketFeatureExtractor] = None, state_path: Optional[Path] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.config, self._engine, self._feed = config, engine, feed
        self._order_client, self._account = order_client, account
        self._feature_extractor = feature_extractor or MarketFeatureExtractor()
        self._state_path: Optional[Path] = Path(state_path) if state_path else (Path(config.state_snapshot_path) if config.state_snapshot_path else None)
        self._on_status = on_status or (lambda line: logger.info(line))
        self._handler: Optional[RosaRojaMarketExecutionHandler] = None
        self._state, self._running, self._shutdown_event = LiveBotState(), False, asyncio.Event()
        self._last_health_check, self._last_broadcast, self._start_time = 0.0, 0.0, time.time()
        self._last_eval_log: float = 0.0
        self._latency_samples: Deque[float] = deque(maxlen=1000)
        self._execution_history: List[ExecutionContext] = []
        self._audit_log_path: Optional[Path] = None
        self._telemetry: Optional[Any] = None
        self._signal_handlers_installed: bool = False

    async def initialize(self) -> None:
        """Inicializa los componentes requeridos."""
        self._feature_extractor = MarketFeatureExtractor(window=200)
        if self._engine is None and self.config.rosa_roja_enabled:
            self._engine = self._create_rosa_roja_engine()
        elif self._engine is None:
            raise ValueError("Engine required but rosa_roja_enabled=False")
        if self._feed is None: self._feed = self._create_feed()
        if self._order_client is None: self._order_client = self._create_order_client()
        if self._account is None: self._account = self._create_account()

        equity = await self._get_equity()
        self._handler = RosaRojaMarketExecutionHandler(
            broker_client=self._order_client, account_equity=equity, symbol=self.config.symbol,
            lot_size=self.config.lot_size, min_qty=self.config.min_lot_size, max_position_pct=self.config.max_position_pct,
        )
        setattr(self._handler, "get_reference_price_callback", self._get_current_mid)
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
        delta_state, delta_time = self._feature_extractor.process(obs)
        if self._engine is None:
            return
        plan = self._engine.process_event(delta_state, delta_time)
        self._state.last_phi_moe = plan.global_confidence
        self._state.last_phi_ritmo = plan.chosen_trajectory.coherence_score if plan.chosen_trajectory else 0.0

        scores: Dict[str, Any] = {}
        if plan.envelope and plan.envelope.metadata and "decision_trace" in plan.envelope.metadata:
            dt = plan.envelope.metadata["decision_trace"]
            self._state.last_lambda_t = float(dt.get("lambda_t", 0.0))
            self._state.last_phi_ritmo = float(dt.get("phi_ritmo", self._state.last_phi_ritmo))
            scores = dt.get("expert_confidences", {})
            self._state.last_expert_votes = [
                {"name": n, "vote": round(float(s) * 2.0 - 1.0, 2), "weight": 1.0, "confidence": round(float(s), 2)}
                for n, s in scores.items()
            ]

        now = time.time()
        if now - self._last_eval_log >= 5.0 or plan.action == "EXECUTE":
            self._last_eval_log = now
            reason = plan.veto_details.get("reason", "") if getattr(plan, "veto_details", None) else ""
            sc_str = " ".join(f"{k.split('_')[0]}:{float(v):.2f}" for k, v in scores.items()) if scores else ""
            logger.info(
                "Evaluation [%s] action=%s reason='%s' mid=%.2f phi_moe=%.3f trades=%d %s",
                self.config.symbol, plan.action, reason, self._get_current_mid(),
                self._state.last_phi_moe, self._state.trades_count, f"[{sc_str}]" if sc_str else "",
            )

        if not self._can_execute(plan):
            return

        if self._handler and await self._handler.dispatch_execution(plan) and plan.action == "EXECUTE":
            self._state.last_execution_time = time.time()
            self._state.last_execution_price = self._get_current_mid()
            self._state.trades_count += 1
            await self._log_execution(plan)
            await self._save_state()
        await self._maybe_health_check()

    def _can_execute(self, plan: Any) -> bool: return can_execute(plan, self.config, self._state, self._get_current_mid())
    def _get_current_mid(self) -> float: return get_current_mid(self._feed)
    async def _log_execution(self, plan: Any) -> None: await log_execution(plan, self._state, self.config, self._audit_log_path, self._execution_history)
    async def _save_state(self) -> None: await save_state(self._state, self._state_path)
    async def _load_state(self) -> None: await load_state(self._state, self._state_path)
    async def shutdown(self) -> None:
        self._running = False; self._shutdown_event.set()
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
    async def _build_telemetry_state(self) -> Dict[str, Any]:
        return await build_telemetry_state(self._state, self.config, self._feed, self._order_client, self._account, self._latency_samples)
    async def _broadcast_telemetry(self) -> None:
        if self._telemetry and (time.time() - self._last_broadcast >= 0.1):
            self._last_broadcast = time.time()
            try: await self._telemetry.broadcast_state(await self._build_telemetry_state())
            except Exception as e: logger.debug(f"Telemetry broadcast error: {e}")
    async def _maybe_health_check(self) -> None:
        if time.time() - self._last_health_check >= self.config.health_check_interval_sec:
            self._last_health_check = time.time(); await self._health_check()
    async def _health_check(self) -> None:
        await perform_health_check(self._state, self.config, self._feed, self._latency_samples, self._running, getattr(self, "on_health_check", None))
    def get_status_line(self) -> str: return format_status_line(self._state, self._feed, self._start_time)


async def create_live_bot(config: Optional[LiveBotConfig] = None, **kwargs) -> LiveBotRunner:
    """Factory para crear e inicializar LiveBotRunner."""
    runner = LiveBotRunner(config or LiveBotConfig(**kwargs))
    await runner.initialize()
    return runner


async def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="ZENIN Live Bot - Event-driven trading")
    p.add_argument("--symbol", default="BTCUSDT"); p.add_argument("--testnet", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true", default=False); p.add_argument("--config", type=Path)
    a = p.parse_args()
    cfg = LiveBotConfig.from_file(a.config) if a.config else LiveBotConfig(symbol=a.symbol, testnet=a.testnet, dry_run=a.dry_run)
    runner = await create_live_bot(cfg)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())