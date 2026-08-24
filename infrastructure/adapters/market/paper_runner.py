"""Trading MVP 0.1 — PaperBotRunner: ZENIN mirando un mercado real.

Ciclo STATELESS con MySQL como estado (crash-proof trivial):

    poll velas cerradas → ventana reciente → ReplayEngine completo
    → upsert idempotente de predicciones → evidencia+gate de la vela nueva
    → OutcomeResolver sobre pendientes vencidos → status line

Regla anti-envenenamiento: el engine al terminar invalida ("feed_ended")
todo lo no vencido dentro de su ventana; esas filas NO se persisten. El
ciclo siguiente re-emite la misma predicción (mismo prediction_id) ya
madura, y el upsert la actualiza a su estado terminal real.

Sin aprendizaje adaptativo: el reward se registra, nada retroalimenta al
modelo en vivo (regla FASE 7 mantenida). La calibración se carga de un
artefacto versionado; mientras no exista, UNCALIBRATED ⇒ NO_TRADE.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from iot_machine_learning.domain.entities.market import Candle
from iot_machine_learning.domain.entities.market.calibration.gate import (
    EvidenceGate,
    EvidenceRecord,
    TradeAction,
)
from iot_machine_learning.domain.entities.market.calibration.pipeline import (
    AdaptiveCalibrator,
    CalibratedPredictor,
    export_calibrator_state,
    import_calibrator_state,
    wrap_predictor,
)
from iot_machine_learning.domain.entities.market.prediction.prediction import (
    Prediction,
)
from iot_machine_learning.domain.entities.market.prediction.resolver import (
    OutcomeResolver,
)
from iot_machine_learning.domain.entities.market.prediction.lifecycle import (
    PredictionStatus,
)
from iot_machine_learning.domain.entities.market.replay import (
    LiveClock,
    MarketReplayEngine,
    ReplayEngineConfig,
)
from iot_machine_learning.domain.entities.market.replay.baselines import (
    EmaCrossoverPredictor,
    MomentumPredictor,
    NaivePredictor,
)


__all__ = ["PaperBotConfig", "CycleReport", "make_predictor", "load_calibrator_state", "save_calibrator_state", "PaperBotRunner"]


@dataclass(frozen=True)
class PaperBotConfig:
    """Configuración del experimento paper (todo explícito, nada oculto)."""

    symbol: str = "BTC-USD"
    interval_seconds: int = 60
    horizons_seconds: tuple[int, ...] = (60, 300, 900)
    predictor_name: str = "momentum"
    neutral_margin: float = 0.05
    require_calibrated: bool = True
    window_candles: int = 180

    def __post_init__(self) -> None:
        if not self.horizons_seconds:
            raise ValueError("horizons no puede ser vacío")
        span = self.window_candles * self.interval_seconds
        needed = max(self.horizons_seconds) + 2 * self.interval_seconds
        if span <= needed:
            raise ValueError(
                f"ventana insuficiente: {self.window_candles} velas × "
                f"{self.interval_seconds}s cubre {span}s pero el horizonte "
                f"máximo {max(self.horizons_seconds)}s necesita > {needed}s"
            )


def make_predictor(name: str):
    """Predictor crudo por nombre (los tres baselines del replay)."""
    predictors = {
        "naive": NaivePredictor,
        "momentum": MomentumPredictor,
        "ema-crossover": EmaCrossoverPredictor,
    }
    if name not in predictors:
        raise ValueError(f"predictor desconocido: {name!r} ({sorted(predictors)})")
    return predictors[name]()


def load_calibrator_state(path: Path) -> AdaptiveCalibrator:
    """Carga el artefacto JSON del calibrador aceptado offline."""
    state = json.loads(path.read_text(encoding="utf-8"))
    return import_calibrator_state(state)


def save_calibrator_state(calibrator: AdaptiveCalibrator, path: Path) -> None:
    """Exporta el artefacto JSON (post-refit offline aceptado)."""
    path.write_text(
        json.dumps(export_calibrator_state(calibrator), indent=2),
        encoding="utf-8",
    )


class PredictionRepoProtocol(Protocol):
    """Subconjunto del contrato de MarketPredictionRepository que usa el runner."""

    def save_batch(self, predictions) -> int: ...

    def pending_outcomes(self, *, symbol: str | None = None): ...


class EvidenceRepoProtocol(Protocol):
    """Subconjunto del contrato de CalibrationEvidenceRepository."""

    def save_batch(self, records) -> int: ...


class FeedProtocol(Protocol):
    """Subconjunto del contrato de BinanceKlinesFeed."""

    def poll_closed(self) -> tuple[Candle, ...]: ...

    def recent_candles(self, limit: int | None = None) -> tuple[Candle, ...]: ...

    def last_close(self, at_or_before: float) -> float | None: ...

    @property
    def connected(self) -> bool: ...


class _StaticCandleFeed:
    """Feed mínimo sobre una tupla de velas (contrato HistoricalFeed)."""

    def __init__(self, candles: tuple[Candle, ...]) -> None:
        self._candles = candles

    def iter_events(self):
        yield from self._candles


@dataclass
class CycleReport:
    """Resultado de un ciclo, para status line y tests."""

    cycle: int
    new_candles: int
    predictions_persisted: int
    evidence_persisted: int
    resolved: int
    waiting: int
    actions: Counter = field(default_factory=Counter)
    latest_prob_raw: float | None = None
    latest_prob_calibrated: float | None = None
    idle: bool = False

    def status_line(self, *, uptime_s: float, calibrator_version: str | None,
                    connected: bool, gaps: int, errors: int) -> str:
        mm, ss = divmod(int(uptime_s), 60)
        hh, mm = divmod(mm, 60)
        total = sum(self.actions.values())
        parts = [
            f"up={hh:02d}:{mm:02d}:{ss:02d}",
            f"cyc={self.cycle}",
            f"velas_nuevas={self.new_candles}",
            f"pred={self.predictions_persisted}",
            f"ev={self.evidence_persisted}",
        ]
        if total:
            rates = " ".join(
                f"{action.value}={100 * n / total:.0f}%"
                for action, n in sorted(self.actions.items())
            )
            parts.append(rates)
        if self.latest_prob_calibrated is not None:
            parts.append(f"P={self.latest_prob_calibrated:.3f}")
        else:
            parts.append("P=-")
        parts.append(f"resueltas={self.resolved}")
        parts.append(f"esperando={self.waiting}")
        parts.append(f"cal={calibrator_version or 'UNCALIBRATED'}")
        parts.append("conn=" + ("OK" if connected else "DEGRADED"))
        parts.append(f"gaps={gaps}")
        parts.append(f"err={errors}")
        return " ".join(parts)


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
        self.cycle_count = 0
        self.totals: Counter[str] = Counter()

    @property
    def calibrator_version(self) -> str | None:
        return self.wrapper._calibrator.get_version()  # noqa: SLF001

    def run_cycle(self) -> CycleReport:
        """Un ciclo completo del MVP (las 8 responsabilidades del usuario)."""
        self.cycle_count += 1
        new_candles = self.feed.poll_closed()
        report = CycleReport(cycle=self.cycle_count, new_candles=len(new_candles), predictions_persisted=0, evidence_persisted=0, resolved=0, waiting=0)
        if not new_candles:
            report.idle = True
            return report

        window_candles = self.feed.recent_candles(limit=self.config.window_candles)
        static_feed = _StaticCandleFeed(window_candles)
        engine_cfg = ReplayEngineConfig(
            symbol=self.config.symbol,
            feed=static_feed,
            interval_seconds=self.config.interval_seconds,
            horizons_seconds=self.config.horizons_seconds,
            predictor=self.wrapper,
            strategy="baseline",
            # El reloj arranca al inicio de la ventana: el engine avanza por
            # evento y un arranque posterior lanzaría ClockRollbackError.
            initial_clock=LiveClock(now=window_candles[0].timestamp),
        )
        result = MarketReplayEngine(engine_cfg).run()

        # Regla anti-envenenamiento + señal accionable:
        # - Las feed_ended NO se persisten (maduran en ciclos futuros y el
        #   upsert las actualizará cuando se re-emitan resueltas).
        # - EXCEPCIÓN: las de la vela más nueva son LA señal en vivo; se
        #   persisten y gatean ahora mismo aunque su horizonte venza después
        #   del fin de ventana (transitoriamente feed_ended).
        # El engine emite la señal de una vela cuando llega la SIGUIENTE
        # (predice sobre last_closed antes de incorporar el evento actual):
        # la observación más reciente CON señal es la penúltima vela.
        signal_ts = (
            window_candles[-2].timestamp if len(window_candles) >= 2 else None
        )
        persistable = [
            p for p in result.predictions
            if not (
                p.status is PredictionStatus.INVALIDATED
                and p.invalidation_reason == "feed_ended"
            )
            or p.observation.timestamp == signal_ts
        ]
        fresh = [
            p for p in result.predictions
            if p.observation.timestamp == signal_ts
        ]
        evidence_by_id = {ev.prediction_id: ev for ev in self.wrapper.evidence_log}

        records: list[EvidenceRecord] = []
        for pred in fresh:
            ev = evidence_by_id.get(pred.prediction_id)
            if ev is None:
                continue  # defensivo: jamás inventar evidencia
            decision = self.gate.decide(ev)
            records.append(EvidenceRecord(evidence=ev, decision=decision))
            self.totals[decision.action.value] += 1
            report.actions[decision.action] += 1
        if fresh and records:
            last = records[-1]
            report.latest_prob_raw = last.evidence.prob_raw
            report.latest_prob_calibrated = last.evidence.prob_calibrated
        elif fresh:
            # Señal fresca sin wrapper-evidencia solo es posible si el gate
            # corre sin calibrador y el log rotó: reportar P como la cruda.
            last_pred = max(fresh, key=lambda p: p.horizon_seconds)
            report.latest_prob_raw = last_pred.probability_up

        prediction_repo = self._prediction_repo_factory()
        report.predictions_persisted = prediction_repo.save_batch(persistable)

        evidence_repo = self._evidence_repo_factory()
        report.evidence_persisted = evidence_repo.save_batch(records)

        resolver = OutcomeResolver()
        pending = prediction_repo.pending_outcomes(symbol=self.config.symbol)
        batch = resolver.resolve(
            (
                row_to_prediction_safe(row)
                for row in pending
            ),
            self.feed,
        )
        resolved = list(batch.resolved)
        if resolved:
            prediction_repo.save_batch(resolved)
        report.resolved = batch.resolved_count
        report.waiting = batch.waiting_count

        line = report.status_line(
            uptime_s=time.monotonic() - self._started_monotonic,
            calibrator_version=self.calibrator_version,
            connected=self.feed.connected,
            gaps=getattr(self.feed, "gaps", 0),
            errors=getattr(self.feed, "errors", 0),
        )
        self._on_status(line)
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
            elapsed = time.monotonic() - started
            sleep_fn(max(1.0, period - elapsed))


def row_to_prediction_safe(row: Any) -> Prediction:
    """Wrapper del mapper del repo para mantener este módulo legible."""
    from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_prediction_repository import (
        row_to_prediction,
    )

    return row_to_prediction(row)
