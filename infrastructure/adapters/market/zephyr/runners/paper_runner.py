"""Trading MVP 0.1 — PaperBotRunner: ZENIN mirando un mercado real.

Ciclo STATELESS con MySQL como estado (crash-proof trivial).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import time
from typing import TYPE_CHECKING, Callable

from iot_machine_learning.domain.entities.market.calibration.gate import EvidenceGate
from iot_machine_learning.domain.entities.market.costs import COST_PROFILES
from iot_machine_learning.domain.entities.market.calibration.pipeline import AdaptiveCalibrator, wrap_predictor
from iot_machine_learning.domain.entities.market.replay import LiveClock, MarketReplayEngine, ReplayEngineConfig
from iot_machine_learning.infrastructure.adapters.market.zephyr.runners.paper_config import (
    PaperBotConfig, load_calibrator_state, make_predictor, save_calibrator_state,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.runners.paper_cycle import (
    build_evidence_records, persist_predictions, resolve_pending,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.runners.paper_protocols import (
    _StaticCandleFeed,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.runners.paper_report import (
    CycleReport,
)

if TYPE_CHECKING:
    from iot_machine_learning.infrastructure.adapters.market.zephyr.runners.paper_protocols import (
        EvidenceRepoProtocol,
        FeedProtocol,
        PredictionRepoProtocol,
    )

__all__ = [
    "PaperBotConfig",
    "CycleReport",
    "make_predictor",
    "load_calibrator_state",
    "save_calibrator_state",
    "PaperBotRunner",
]


class PaperBotRunner:
    """Orquestador del paper bot. Sin dinero, sin aprendizaje, sin humo."""

    def __init__(
        self,
        *,
        config: PaperBotConfig,
        feed: FeedProtocol,
        prediction_repo_factory: Callable[[], PredictionRepoProtocol],
        evidence_repo_factory: Callable[[], EvidenceRepoProtocol],
        calibrator_state_path: Path | None = None,
        on_status: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.feed = feed
        self._prediction_repo_factory = prediction_repo_factory
        self._evidence_repo_factory = evidence_repo_factory
        self._on_status = on_status or (lambda line: print(line, flush=True))
        self._clock = clock
        self._started_monotonic = time.monotonic()

        raw = make_predictor(config.predictor_name)
        if calibrator_state_path is not None and Path(calibrator_state_path).exists():
            calibrator = load_calibrator_state(Path(calibrator_state_path))
        else:
            calibrator = AdaptiveCalibrator()
        self.wrapper = wrap_predictor(raw, calibrator)
        self.gate = EvidenceGate(
            neutral_margin=config.neutral_margin,
            require_calibrated=config.require_calibrated,
        )
        try:
            self.cost_model = COST_PROFILES[config.symbol]
        except KeyError:
            raise ValueError(
                f"sin perfil de costos para {config.symbol!r} "
                f"(conocidos: {sorted(COST_PROFILES)}): el gate económico "
                "falla cerrado, no abierto"
            ) from None
        self.cycle_count = 0
        self.totals: Counter[str] = Counter()

    @property
    def calibrator_version(self) -> str | None:
        return self.wrapper._calibrator.get_version()  # noqa: SLF001

    def run_cycle(self) -> CycleReport:
        """Un ciclo completo del MVP (las 8 responsabilidades del usuario)."""
        self.cycle_count += 1
        new_candles = self.feed.poll_closed()
        report = CycleReport(
            cycle=self.cycle_count,
            new_candles=len(new_candles),
            predictions_persisted=0,
            evidence_persisted=0,
            resolved=0,
            waiting=0,
        )
        if not new_candles:
            report.idle = True
            return report

        window_candles = self.feed.recent_candles(limit=self.config.window_candles)
        signal_ts = window_candles[-2].timestamp if len(window_candles) >= 2 else None
        result = MarketReplayEngine(
            ReplayEngineConfig(
                symbol=self.config.symbol,
                feed=_StaticCandleFeed(window_candles),
                interval_seconds=self.config.interval_seconds,
                horizons_seconds=self.config.horizons_seconds,
                predictor=self.wrapper,
                strategy="baseline",
                initial_clock=LiveClock(now=window_candles[0].timestamp),
            )
        ).run()

        records = build_evidence_records(
            result, signal_ts, report,
            gate=self.gate, cost_model=self.cost_model,
            evidence_log=self.wrapper.evidence_log,
            require_positive_net=self.config.require_positive_net,
            totals=self.totals,
        )
        persist_predictions(
            result, signal_ts, records, report,
            prediction_repo_factory=self._prediction_repo_factory,
            evidence_repo_factory=self._evidence_repo_factory,
        )
        resolve_pending(
            report,
            symbol=self.config.symbol,
            prediction_repo_factory=self._prediction_repo_factory,
            feed=self.feed,
        )
        self._on_status(
            report.status_line(
                uptime_s=time.monotonic() - self._started_monotonic,
                calibrator_version=self.calibrator_version,
                connected=self.feed.connected,
                gaps=getattr(self.feed, "gaps", 0),
                errors=getattr(self.feed, "errors", 0),
            )
        )
        return report

    def run(
        self,
        *,
        max_cycles: int | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        stop: Callable[[], bool] | None = None,
    ) -> None:
        """Loop continuo; para por max_cycles, stop() o KeyboardInterrupt."""
        period = self.config.interval_seconds
        while True:
            started = time.monotonic()
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                raise
            if max_cycles is not None and self.cycle_count >= max_cycles:
                return
            if stop is not None and stop():
                return
            sleep_fn(max(1.0, period - (time.monotonic() - started)))
