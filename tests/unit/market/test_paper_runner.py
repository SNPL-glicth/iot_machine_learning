"""Trading MVP 0.1 — PaperBotRunner con fakes (sin red, sin MySQL)."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from iot_machine_learning.domain.entities.market import Candle, DataStatus
from iot_machine_learning.domain.entities.market.calibration import (
    AdaptiveCalibrator,
    CalibrationMethod,
    ContextKey,
    TradeAction,
    export_calibrator_state,
    try_refit,
)
from iot_machine_learning.infrastructure.adapters.market.paper_runner import (
    PaperBotConfig,
    PaperBotRunner,
)


# ─── Fakes ──────────────────────────────────────────────────────────────────


def _candle(ts: float, close: float) -> Candle:
    return Candle(
        symbol="BTC-USD",
        timestamp=ts,
        data_status=DataStatus.REALTIME,
        source_provider="fake",
        interval_seconds=60,
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=1.0,
    )


class FakeFeed:
    """Feed determinista: suelta velas programadas por poll."""

    def __init__(self, script: list[list[Candle]]) -> None:
        self.script = [list(p) for p in script]
        self.buffer: list[Candle] = []
        self.gaps = 0
        self.errors = 0

    def poll_closed(self):
        page = self.script.pop(0) if self.script else []
        self.buffer.extend(page)
        return tuple(page)

    def recent_candles(self, limit=None):
        window = tuple(self.buffer)
        return window[-limit:] if limit else window

    def last_close(self, at_or_before: float):
        best = None
        for c in self.buffer:
            if c.timestamp <= at_or_before:
                best = c.close
        return best

    @property
    def connected(self):
        return True


class FakePredictionRepo:
    def __init__(self):
        self.saved: list = []

    def save_batch(self, predictions):
        self.saved.extend(predictions)
        return len(predictions)

    def pending_outcomes(self, *, symbol=None):
        return []


class FakeEvidenceRepo:
    def __init__(self):
        self.records: list = []

    def save_batch(self, records):
        self.records.extend(records)
        return len(records)


def _seed_window(start_ts: float, n: int, base: float = 100.0) -> list[Candle]:
    rng = random.Random(5)
    out = []
    price = base
    for i in range(n):
        price *= 1.0 + rng.uniform(-0.002, 0.004)
        out.append(_candle(start_ts + i * 60, round(price, 2)))
    return out


def _runner(script, **overrides):
    config = overrides.pop("config", PaperBotConfig())
    feed = FakeFeed(script)
    pred_repo = overrides.pop("pred_repo", FakePredictionRepo())
    ev_repo = overrides.pop("ev_repo", FakeEvidenceRepo())
    statuses: list[str] = []
    runner = PaperBotRunner(
        config=config,
        feed=feed,
        prediction_repo_factory=lambda: pred_repo,
        evidence_repo_factory=lambda: ev_repo,
        on_status=statuses.append,
        **overrides,
    )
    return runner, feed, pred_repo, ev_repo, statuses


CONFIG = PaperBotConfig(window_candles=40, horizons_seconds=(60,))


class TestRunCycle:
    def test_ciclo_persiste_predicciones_y_evidencia_solo_de_vela_nueva(self):
        seed = _seed_window(3600.0, CONFIG.window_candles)
        runner, feed, pred_repo, ev_repo, statuses = _runner(
            [seed], config=CONFIG
        )
        report = runner.run_cycle()

        # Persistieron predicciones (menos las feed_ended históricas).
        assert report.predictions_persisted > 0
        assert len(pred_repo.saved) == report.predictions_persisted
        signal_ts = seed[-2].timestamp
        # Ninguna fila envenenada: feed_ended solo se tolera para la señal
        # en vivo (penúltima vela; el siguiente ciclo la actualiza).
        assert not any(
            p.status.value == "invalidated"
            and p.invalidation_reason == "feed_ended"
            and p.observation.timestamp != signal_ts
            for p in pred_repo.saved
        )
        # Evidencia SOLO para la vela más nueva (una señal por horizonte=1).
        assert report.evidence_persisted == 1
        assert len(ev_repo.records) == 1
        record = ev_repo.records[0]
        assert record.evidence.observation_timestamp == seed[-2].timestamp
        # Sin artefacto de calibrador ⇒ UNCALIBRATED ⇒ NO_TRADE honesto.
        assert record.decision.action is TradeAction.NO_TRADE
        assert "NO_TRADE=100%" in statuses[-1]
        assert "cal=UNCALIBRATED" in statuses[-1]

    def test_ciclo_sin_velas_nuevas_es_idle_y_no_persiste(self):
        runner, *_ = _runner([[]], config=CONFIG)
        report = runner.run_cycle()
        assert report.idle is True
        assert report.predictions_persisted == 0
        assert report.evidence_persisted == 0

    def test_segundo_ciclo_no_duplica_evidencia_por_misma_senal(self):
        seed = _seed_window(3600.0, CONFIG.window_candles + 5)
        split = CONFIG.window_candles - 1
        runner, feed, _, ev_repo, _ = _runner(
            [seed[:split], seed[split:]], config=CONFIG
        )
        runner.run_cycle()
        first_ids = {r.evidence.prediction_id for r in ev_repo.records}
        runner.run_cycle()
        second_ids = {
            r.evidence.prediction_id for r in ev_repo.records
        } - first_ids
        # La vela nueva del ciclo 2 produce IDs distintos a los del ciclo 1.
        assert first_ids.isdisjoint(second_ids)

    def test_con_artefacto_calibrado_la_version_fluye_al_status(self, tmp_path: Path):
        rng = random.Random(7)
        ctx = ContextKey("momentum", 900, "ALL")
        pairs = [(ctx, 0.90, rng.random() < 0.50) for _ in range(600)]
        calibrator = AdaptiveCalibrator(method=CalibrationMethod.PLATT)
        calibrator.set_version("v4")
        assert try_refit(calibrator, pairs) is True
        artifact = tmp_path / "calibrator.json"
        artifact.write_text(__import__("json").dumps(export_calibrator_state(calibrator)))

        seed = _seed_window(3600.0, CONFIG.window_candles)
        runner, _, _, _, statuses = _runner(
            [seed], config=CONFIG, calibrator_state_path=artifact
        )
        report = runner.run_cycle()
        assert "cal=v4" in statuses[-1]
        # El fallback activo recalibró: la evidencia ya no es UNCALIBRATED.
        assert all(
            ev.fallback_level != "UNCALIBRATED"
            for ev in runner.wrapper.evidence_log
        )


class TestRunLoop:
    def test_run_para_por_max_cycles_y_duerme_entre_ciclos(self):
        seed = _seed_window(3600.0, CONFIG.window_candles + 3)
        sleeps: list[float] = []
        runner, feed, _, _, _ = _runner(
            [seed[:40], seed[40:], [], []], config=CONFIG
        )
        runner.run(max_cycles=3, sleep_fn=sleeps.append)
        assert runner.cycle_count == 3
        # Duerme ENTRE ciclos: tras el último no hay espera.
        assert len(sleeps) == 2
        assert all(s >= 1.0 for s in sleeps)

    def test_stop_callback_corta_el_loop(self):
        runner, *_ = _runner([[]], config=CONFIG)
        calls = {"n": 0}

        def stop():
            calls["n"] += 1
            return True

        runner.run(max_cycles=None, sleep_fn=lambda _: None, stop=stop)
        assert runner.cycle_count == 1
        assert calls["n"] == 1
