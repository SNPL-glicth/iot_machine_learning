"""LiveBotRunner -- Loop principal event-driven para trading live.

Arquitectura:
    BinanceWSFeed (async) -> MarketFeatureExtractor -> RosaRojaEngine
                                                    |
                                                    v
                                         RosaRojaMarketExecutionHandler
                                                    |
                                                    v
                                         BinanceOrderClient (REST)

Flujo por tick:
1. Recibir observación (Quote, Trade, OrderBookSnapshot)
2. Extraer features -> delta_state vector (10 dims) + delta_time
3. RosaRojaEngine.process_event(delta_state, delta_time) -> ExecutionPlan
4. Verificar cooldown/hysteresis
5. ExecutionHandler.dispatch_execution() -> órdenes en Binance
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator, Callable, Deque, Dict, List, Optional, Set
from collections import defaultdict

import numpy as np

from iot_machine_learning.domain.entities.market.observations import MarketObservation
from iot_machine_learning.infrastructure.adapters.market.binance import (
    BinanceWSFeed,
    BinanceWSClient,
    OrderBookL2,
    OrderBookMetrics,
    FeedStats,
)
from iot_machine_learning.infrastructure.adapters.market.telemetry_server import (
    TelemetryBroadcaster,
    create_telemetry_server,
)
from iot_machine_learning.infrastructure.adapters.market.rosa_roja_features import MarketFeatureExtractor
from iot_machine_learning.core.orchestration.rosa_roja.engine import RosaRojaEngine
from iot_machine_learning.infrastructure.adapters.market.rosa_roja_market_handler import (
    RosaRojaMarketExecutionHandler,
    BrokerClientProtocol,
)
from iot_machine_learning.core.orchestration.rosa_roja.domain.execution import (
    ExecutionPlan,
    ActionEnvelope,
)
from iot_machine_learning.infrastructure.adapters.market.live_config import LiveBotConfig
from iot_machine_learning.infrastructure.adapters.market.binance.order_client import BinanceOrderClient
from iot_machine_learning.infrastructure.adapters.market.binance.account import BinanceAccount

logger = logging.getLogger(__name__)


@dataclass
class LiveBotState:
    """Estado interno del bot para persistencia y recuperación."""
    cycle_count: int = 0
    last_execution_time: float = 0.0
    last_execution_price: float = 0.0
    last_execution_side: str = ""
    last_phi_moe: float = 0.0
    last_lambda_t: float = 0.0
    last_phi_ritmo: float = 0.0
    active_orders: Dict[str, Dict] = field(default_factory=dict)
    current_position: float = 0.0  # BTC, positivo=long, negativo=short
    total_pnl: float = 0.0
    trades_count: int = 0
    last_error: Optional[str] = None


@dataclass
class ExecutionContext:
    """Contexto de una ejecución para auditoría."""
    timestamp: float
    phi_moe: float
    lambda_t: float
    phi_ritmo: float
    action: str
    side: str
    qty: float
    price: float
    order_type: str
    decision_trace: Dict
    telemetry_hash: str


class LiveBotRunner:
    """
    Runner principal del bot live.

    Responsabilidades:
    - Ciclo de vida: connect -> run loop -> graceful shutdown
    - Gestión de estado y persistencia
    - Cooldown / hysteresis / risk checks
    - Métricas y health checks
    - Graceful shutdown (SIGTERM/SIGINT)
    """

    def __init__(
        self,
        config: LiveBotConfig,
        *,
        engine: Optional[RosaRojaEngine] = None,
        feed: Optional["BinanceWSFeed"] = None,
        order_client: Optional["BinanceOrderClient"] = None,
        account: Optional["BinanceAccount"] = None,
        feature_extractor: Optional["MarketFeatureExtractor"] = None,
        state_path: Optional[Path] = None,
        on_status: Optional[Callable[[str], None]] = None,
    ):
        self.config = config
        self._engine = engine
        self._feed = feed
        self._order_client = order_client
        self._account = account
        self._feature_extractor = feature_extractor or MarketFeatureExtractor()
        self._state_path = state_path or config.state_snapshot_path
        self._on_status = on_status or (lambda line: logger.info(line))

        # Componentes lazy-loaded
        self._feed: Optional["BinanceWSFeed"] = feed
        self._engine: Optional[RosaRojaEngine] = engine
        self._order_client: Optional["BinanceOrderClient"] = order_client
        self._account: Optional["BinanceAccount"] = account
        self._handler: Optional["RosaRojaMarketExecutionHandler"] = None

        # Estado interno
        self._state = LiveBotState()
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._last_health_check = 0.0
        self._latency_samples: Deque[float] = deque(maxlen=1000)
        self._execution_history: List[ExecutionContext] = []
        self._audit_log_path: Optional[Path] = None

        # Telemetry broadcaster for TUI
        self._telemetry: Optional[TelemetryBroadcaster] = None

        # Signal handlers
        self._signal_handlers_installed = False

    async def initialize(self) -> None:
        """Inicializa todos los componentes."""
        logger.info("Initializing LiveBotRunner", extra={"symbol": self.config.symbol})

        # 1. Feature extractor
        self._feature_extractor = MarketFeatureExtractor(window=200)

        # 2. Rosa Roja Engine
        if self._engine is None:
            if self.config.rosa_roja_enabled:
                self._engine = self._create_rosa_roja_engine()
            else:
                raise ValueError("Engine required but rosa_roja_enabled=False")

        # 3. Feed
        if self._feed is None:
            self._feed = BinanceWSFeed(
                symbol=self.config.symbol,
                testnet=self.config.testnet,
                depth_speed=self.config.depth_speed,
                include_trades=self.config.include_trades,
                include_book_ticker=self.config.include_book_ticker,
                include_kline=self.config.include_kline,
                kline_interval=self.config.kline_interval,
                max_queue_size=self.config.ws_max_queue_size,
                snapshot_interval_sec=self.config.ob_snapshot_interval_sec,
                on_observation=self._on_observation_callback,
                on_metrics=self._on_feed_metrics,
                on_state_change=self._on_feed_state_change,
            )

        # 4. Order client & account
        if self._order_client is None:
            self._order_client = BinanceOrderClient(
                api_key=self._get_api_key(),
                api_secret=self._get_api_secret(),
                testnet=self.config.testnet,
            )

        if self._account is None:
            self._account = BinanceAccount(
                client=self._order_client,
                symbol=self.config.symbol,
            )

        # 5. Execution handler
        self._handler = RosaRojaMarketExecutionHandler(
            broker_client=self._order_client,
            account_equity=await self._get_equity(),
            symbol=self.config.symbol,
            lot_size=self.config.lot_size,
            min_qty=self.config.min_lot_size,
            max_position_pct=self.config.max_position_pct,
        )

        # 6. Audit log
        if self.config.enable_audit_log and self.config.audit_log_path:
            self._audit_log_path = Path(self.config.audit_log_path)
            self._audit_log_path.mkdir(parents=True, exist_ok=True)

        # 7. Load state
        await self._load_state()

        # 8. Install signal handlers
        self._install_signal_handlers()

        # 9. Initialize telemetry broadcaster for TUI
        if self.config.enable_metrics_export:
            self._telemetry = await create_telemetry_server(self)

        logger.info("LiveBotRunner initialized successfully", extra={"symbol": self.config.symbol})

    def _create_rosa_roja_engine(self) -> RosaRojaEngine:
        """Crea engine Rosa Roja con configuración por defecto."""
        from iot_machine_learning.core.orchestration.rosa_roja.modules.module1_ingestion import MahalanobisFilter
        from iot_machine_learning.core.orchestration.rosa_roja.modules.rhythm_generator import RhythmTrajectoryGenerator
        from iot_machine_learning.core.orchestration.rosa_roja.modules.module3_moe_gating import MultiplicativeMoEGating
        from iot_machine_learning.infrastructure.ml.adapters import (
            TaylorExpertAdapter,
            KalmanExpertAdapter,
            StatisticalExpertAdapter,
        )
        from iot_machine_learning.infrastructure.ml.engines.taylor.engine import TaylorPredictionEngine
        from iot_machine_learning.infrastructure.ml.engines.kalman.engine import KalmanPredictionEngine
        from iot_machine_learning.infrastructure.ml.engines.statistical.engine import StatisticalPredictionEngine

        ingestion = MahalanobisFilter(
            noise_threshold=3.0,
            history_window=100,
            min_samples_for_cov=20,
        )
        rhythm = RhythmTrajectoryGenerator(
            min_trajectory_len=11,
            max_trajectory_len=15,
            top_k=4,
            oversample_factor=2,
            max_random_walk_steps=40,
        )
        gating = MultiplicativeMoEGating(variance_penalty=0.5)

        # Expertos
        taylor_engine = TaylorPredictionEngine()
        kalman_engine = KalmanPredictionEngine()
        stat_engine = StatisticalPredictionEngine()

        taylor_expert = TaylorExpertAdapter(engine=taylor_engine)
        kalman_expert = KalmanExpertAdapter(engine=kalman_engine)
        stat_expert = StatisticalExpertAdapter(engine=stat_engine)

        return RosaRojaEngine(
            ingestion_filter=ingestion,
            rhythm_generator=rhythm,
            moe_gating=gating,
            expert_jury=[taylor_expert, kalman_expert, stat_expert],
            drift_sensors=[],
            outlier_reset_threshold=3,
            exploration_boost_events=5,
        )

    def _get_api_key(self) -> str:
        import os
        key = os.getenv("BINANCE_API_KEY") or os.getenv("BINANCE_TESTNET_API_KEY")
        if not key:
            raise ValueError("BINANCE_API_KEY or BINANCE_TESTNET_API_KEY not set")
        return key

    def _get_api_secret(self) -> str:
        import os
        secret = os.getenv("BINANCE_API_SECRET") or os.getenv("BINANCE_TESTNET_API_SECRET")
        if not secret:
            raise ValueError("BINANCE_API_SECRET or BINANCE_TESTNET_API_SECRET not set")
        return secret

    async def _get_equity(self) -> float:
        """Obtiene equity actual de la cuenta."""
        if self._account:
            return await self._account.get_equity()
        return 10000.0  # Default para dry-run

    # Callbacks del feed
    def _on_observation_callback(self, obs) -> None:
        """Callback interno para procesar observación en el loop principal."""
        pass  # Se maneja en el loop principal

    def _on_feed_metrics(self, metrics: dict) -> None:
        pass

    def _on_feed_state_change(self, old, new) -> None:
        logger.info("Feed state change", extra={"from": old.value, "to": new.value, "symbol": self.config.symbol})

    # Main loop
    async def run(self) -> None:
        """Loop principal event-driven."""
        if not self._feed or not self._engine:
            raise RuntimeError("Runner not initialized. Call initialize() first.")

        self._running = True
        await self._feed.connect()

        logger.info("Starting live trading loop", extra={"symbol": self.config.symbol})

        try:
            async for obs in self._feed.iter_observations():
                if not self._running:
                    break

                # Health check periódico
                await self._maybe_health_check()

                # Check shutdown
                if self._shutdown_event.is_set():
                    break

                # Procesar observación
                await self._process_observation(obs)

                # Broadcast telemetry to TUI
                if self._telemetry:
                    await self._broadcast_telemetry()

        except asyncio.CancelledError:
            logger.info("Runner cancelled")
        except Exception as e:
            logger.error("Runner error", extra={"error": str(e)})
            self._state.last_error = str(e)
            raise
        finally:
            await self.shutdown()

    async def _build_telemetry_state(self) -> Dict[str, Any]:
        """Construye el estado de telemetría para el TUI."""
        ob = self._feed.order_book if self._feed else None
        return {
            "timestamp": time.time(),
            "symbol": self.config.symbol,
            "mode": "TESTNET" if self.config.testnet else "MAINNET",
            "latency_p50_ms": np.percentile(self._latency_samples, 50) if self._latency_samples else 0.0,
            "phi_moe": self._state.last_phi_moe,
            "lambda_t": self._state.last_lambda_t,
            "phi_ritmo": self._state.last_phi_ritmo,
            "best_bid": self._feed.order_book.best_bid if self._feed and self._feed.order_book else 0.0,
            "best_ask": self._feed.order_book.best_ask if self._feed and self._feed.order_book else 0.0,
            "bid_vol": self._feed.order_book.metrics.bid_volume if self._feed and self._feed.order_book.metrics else 0.0,
            "ask_vol": self._feed.order_book.metrics.ask_volume if self._feed and self._feed.order_book.metrics else 0.0,
            "obi": self._feed.order_book.metrics.volume_imbalance if self._feed and self._feed.order_book.metrics else 0.0,
            "microprice": self._feed.order_book.metrics.microprice if self._feed and self._feed.order_book.metrics else 0.0,
            "experts": [],  # TODO: poblar desde MoE
            "position_qty": self._state.current_position,
            "entry_price": self._state.last_execution_price,
            "pnl_usd": self._state.total_pnl,
            "pnl_pct": 0.0,  # TODO: calcular
            "last_action": getattr(self._state, 'last_action', 'HOLD'),
            "last_reason": getattr(self._state, 'last_reason', ''),
        }

    async def _broadcast_telemetry(self) -> None:
        """Envía estado actual al TUI vía WebSocket."""
        if self._telemetry:
            try:
                state = await self._build_telemetry_state()
                await self._telemetry.broadcast_state(state)
            except Exception as e:
                logger.debug(f"Telemetry broadcast error: {e}")

    async def _process_observation(self, obs) -> None:
        """Procesa una observación del mercado."""
        receive_time = time.time()

        # 1. Feature extraction
        delta_state, delta_time = self._feature_extractor.process(obs)

        # 2. Rosa Roja Engine
        plan = self._engine.process_event(delta_state, delta_time)

        # 3. Extraer métricas del plan
        phi_moe = plan.global_confidence
        lambda_t = getattr(plan, 'lambda_t', 0.0) if hasattr(plan, 'lambda_t') else 0.0
        phi_ritmo = getattr(plan, 'phi_ritmo', 0.0) if hasattr(plan, 'phi_ritmo') else 0.0

        # Actualizar estado
        self._state.last_phi_moe = phi_moe
        self._state.last_lambda_t = getattr(plan, 'lambda_t', 0.0) if hasattr(plan, 'lambda_t') else 0.0
        self._state.last_phi_ritmo = getattr(plan, 'phi_ritmo', 0.0) if hasattr(plan, 'phi_ritmo') else 0.0

        # 3. Verificar cooldown / hysteresis
        if not self._can_execute(plan):
            return

        # 4. Ejecutar plan
        if self._handler:
            success = self._handler.dispatch_execution(plan)
            if success and plan.action == "EXECUTE":
                self._state.last_execution_time = time.time()
                self._state.last_execution_price = self._get_current_mid()
                self._state.trades_count += 1
                await self._log_execution(plan)
                await self._save_state()

        # Health check periódico
        await self._maybe_health_check()

    def _can_execute(self, plan) -> bool:
        """Verifica cooldown, hysteresis y risk checks."""
        if plan.action != "EXECUTE":
            return True  # HOLD/FLUSH no requieren cooldown

        now = time.time()
        current_price = self._get_current_mid()

        # Cooldown dinámico
        if self.config.dynamic_cooldown:
            lambda_t = self._state.last_lambda_t
            cooldown_ms = self.config.get_effective_cooldown(self._state.last_lambda_t)
        else:
            cooldown_ms = float(self.config.cooldown_ms)

        # Tiempo mínimo
        if self._state.last_execution_time > 0:
            elapsed_ms = (time.time() - self._state.last_execution_time) * 1000
            if elapsed_ms < cooldown_ms:
                return False

        # Cambio mínimo de precio
        if self._state.last_execution_price > 0:
            price_change_pct = abs(current_price - self._state.last_execution_price) / self._state.last_execution_price
            if price_change_pct < self.config.min_price_change_pct:
                return False

        # Risk checks adicionales
        if self._state.current_position != 0:
            # Ya tenemos posición, verificar riesgo
            if abs(self._state.current_position) >= self.config.max_position_pct:
                return False

        # Phi_MoE threshold
        if self._state.last_phi_moe < self.config.phi_moe_threshold:
            return False

        # Lambda threshold
        if self._state.last_lambda_t >= self.config.emergency_lambda_threshold:
            return False

        return True

    def _get_current_mid(self) -> float:
        """Obtiene mid-price actual del order book."""
        if self._feed and self._feed.order_book.is_initialized:
            return self._feed.order_book.mid_price or 0.0
        return 0.0

    async def _log_execution(self, plan) -> None:
        """Registra ejecución en audit log."""
        if not self.config.enable_audit_log or not self._audit_log_path:
            return

        # Crear contexto de ejecución
        ctx = ExecutionContext(
            timestamp=time.time(),
            phi_moe=self._state.last_phi_moe,
            lambda_t=self._state.last_lambda_t,
            phi_ritmo=self._state.last_phi_ritmo,
            action=plan.action,
            side="",  # Se llena en handler
            qty=0.0,
            price=0.0,
            order_type="",
            decision_trace=plan.envelope.metadata.get("decision_trace", {}) if plan.envelope else {},
            telemetry_hash="",  # Se llena en handler
        )

        self._execution_history.append(ctx)

        # Escribir a audit log (async, no bloqueante)
        if self._audit_log_path:
            log_file = self._audit_log_path / f"audit_{time.strftime('%Y%m%d')}.ndjson"
            try:
                import aiofiles
                async with aiofiles.open(log_file, "a") as f:
                    import json
                    await f.write(json.dumps({
                        "timestamp": ctx.timestamp,
                        "phi_moe": ctx.phi_moe,
                        "lambda_t": ctx.lambda_t,
                        "phi_ritmo": ctx.phi_ritmo,
                        "action": ctx.action,
                    }) + "\n")
            except Exception as e:
                logger.warning("Failed to write audit log", extra={"error": str(e)})

    async def _maybe_health_check(self) -> None:
        """Health check periódico."""
        now = time.time()
        if now - self._last_health_check >= self.config.health_check_interval_sec:
            self._last_health_check = time.time()
            await self._health_check()

    async def _health_check(self) -> None:
        """Health check completo."""
        health = {
            "timestamp": time.time(),
            "symbol": self.config.symbol,
            "running": self._running,
            "feed_connected": self._feed.is_connected if self._feed else False,
            "feed_state": self._feed.state.value if self._feed else "none",
            "order_book_initialized": self._feed.order_book.is_initialized if self._feed else False,
            "order_book_metrics": self._feed.order_book.metrics.to_dict() if self._feed and self._feed.order_book.metrics else None,
            "engine_phi_moe": self._state.last_phi_moe,
            "engine_lambda_t": self._state.last_lambda_t,
            "engine_phi_ritmo": self._state.last_phi_ritmo,
            "position": self._state.current_position,
            "trades_count": self._state.trades_count,
            "latency_p50_ms": np.percentile(self._latency_samples, 50) if self._latency_samples else 0,
            "latency_p99_ms": np.percentile(self._latency_samples, 99) if self._latency_samples else 0,
            "errors": self._state.last_error,
        }

        # Log health
        logger.info("Health check", extra=health)

        # Alertas
        if self._latency_samples and np.percentile(self._latency_samples, 99) > self.config.max_latency_ms:
            logger.warning("High latency detected", extra={"p99_latency_ms": np.percentile(self._latency_samples, 99)})

        if self._feed and not self._feed.is_connected:
            logger.warning("Feed disconnected", extra={"symbol": self.config.symbol})

        # Callback
        if hasattr(self, 'on_health_check'):
            try:
                self.on_health_check(health)
            except Exception as e:
                logger.error("Error in health check callback", extra={"error": str(e)})

    # Persistencia de estado
    async def _save_state(self) -> None:
        if not self._state_path:
            return

        try:
            state = {
                "cycle_count": self._state.cycle_count,
                "last_execution_time": self._state.last_execution_time,
                "last_execution_price": self._state.last_execution_price,
                "last_execution_side": self._state.last_execution_side,
                "last_phi_moe": self._state.last_phi_moe,
                "last_lambda_t": self._state.last_lambda_t,
                "last_phi_ritmo": self._state.last_phi_ritmo,
                "current_position": self._state.current_position,
                "total_pnl": self._state.total_pnl,
                "trades_count": self._state.trades_count,
                "last_error": self._state.last_error,
                "timestamp": time.time(),
            }
            import aiofiles
            import json
            async with aiofiles.open(self._state_path, "w") as f:
                await f.write(json.dumps(state, indent=2))
        except Exception as e:
            logger.warning("Failed to save state", extra={"error": str(e)})

    async def _load_state(self) -> None:
        if not self._state_path or not Path(self._state_path).exists():
            return

        try:
            import aiofiles
            import json
            async with aiofiles.open(self._state_path, "r") as f:
                data = json.loads(await f.read())
                self._state.cycle_count = data.get("cycle_count", 0)
                self._state.last_execution_time = data.get("last_execution_time", 0.0)
                self._state.last_execution_price = data.get("last_execution_price", 0.0)
                self._state.last_execution_side = data.get("last_execution_side", "")
                self._state.last_phi_moe = data.get("last_phi_moe", 0.0)
                self._state.last_lambda_t = data.get("last_lambda_t", 0.0)
                self._state.last_phi_ritmo = data.get("last_phi_ritmo", 0.0)
                self._state.current_position = data.get("current_position", 0.0)
                self._state.total_pnl = data.get("total_pnl", 0.0)
                self._state.trades_count = data.get("trades_count", 0)
                self._state.last_error = data.get("last_error")
            logger.info("State loaded", extra={"cycle": self._state.cycle_count})
        except Exception as e:
            logger.warning("Failed to load state", extra={"error": str(e)})

    # Signal handling
    def _install_signal_handlers(self) -> None:
        if self._signal_handlers_installed:
            return

        def signal_handler(signum, frame):
            logger.info("Signal received, initiating shutdown", extra={"signal": signum})
            self._shutdown_event.set()

        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, signal_handler)

        self._signal_handlers_installed = True

    async def shutdown(self) -> None:
        """Apagado graceful."""
        logger.info("Shutting down LiveBotRunner", extra={"symbol": self.config.symbol})
        self._running = False
        self._shutdown_event.set()

        # 1. Cancelar órdenes abiertas
        if self._handler:
            self._handler.trigger_emergency_flush("Graceful shutdown")

        # 2. Cerrar posición si existe
        if self._state.current_position != 0 and self._order_client:
            try:
                await self._order_client.close_position(self.config.symbol)
            except Exception as e:
                logger.error("Failed to close position on shutdown", extra={"error": str(e)})

        # 3. Desconectar feed
        if self._feed:
            await self._feed.disconnect()

        # 3. Cerrar order client
        if self._order_client:
            await self._order_client.close()

        # 4. Guardar estado final
        await self._save_state()

        logger.info("LiveBotRunner shutdown complete", extra={"symbol": self.config.symbol})

    # Status line para CLI
    def get_status_line(self) -> str:
        uptime = time.time() - getattr(self, '_start_time', time.time())
        hh, rem = divmod(int(uptime), 3600)
        mm, ss = divmod(rem, 60)

        feed = self._feed
        ob = self._feed.order_book if self._feed else None

        parts = [
            f"up={hh:02d}:{mm:02d}:{ss:02d}",
            f"phi={self._state.last_phi_moe:.3f}",
            f"lambda={self._state.last_lambda_t:.3f}",
            f"phi_r={self._state.last_phi_ritmo:.3f}",
            f"pos={self._state.current_position:.6f}",
            f"trades={self._state.trades_count}",
            f"feed={'UP' if self._feed and self._feed.is_connected else 'DOWN'}",
            f"ob={'INIT' if self._feed and self._feed.order_book.is_initialized else 'SYNCING'}",
        ]
        return " | ".join(parts)


# Factory function
async def create_live_bot(
    config: Optional[LiveBotConfig] = None,
    **kwargs,
) -> LiveBotRunner:
    """Factory para crear y inicializar LiveBotRunner."""
    if config is None:
        config = LiveBotConfig(**kwargs)

    runner = LiveBotRunner(config)
    await runner.initialize()
    return runner


# CLI entry point
async def main():
    import argparse

    parser = argparse.ArgumentParser(description="ZENIN Live Bot - Event-driven trading")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--testnet", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--config", type=Path, help="Path to config JSON")
    parser.add_argument("--max-cycles", type=int, help="Max cycles (0 = infinite)")
    args = parser.parse_args()

    config = LiveBotConfig(
        symbol=args.symbol,
        testnet=args.testnet,
        dry_run=args.dry_run,
    )

    if args.config:
        config = LiveBotConfig.from_file(args.config)

    runner = await create_live_bot(config)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())