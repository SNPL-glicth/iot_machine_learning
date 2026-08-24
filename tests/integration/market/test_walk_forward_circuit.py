"""Integration — FASE 9.1: flujo walk-forward real contra MySQL.

Mini WFA sintético (símbolo TEST-WF, prefijo TEST-WF- limpio al final):
TRAIN con outcomes reales → analyzer → proposer → guard (rechaza con
1 día: historial insuficiente) → guard permisivo (acepta) → versión →
TEST evaluado con los pesos de la versión vía ``evaluate_window``.
Cubre la cadena completa que orquesta el script sin tocar el engine
(el engine ya está cubierto por las pruebas de replay).
"""

from __future__ import annotations

import json

import pytest
from iot_machine_learning.domain.entities.market import Candle, DataStatus
from iot_machine_learning.domain.entities.market.adaptation import (
    AdaptationGuard,
    PerformanceAnalyzer,
    WeightProposer,
)
from iot_machine_learning.domain.entities.market.prediction import (
    Prediction,
    PredictionInterval,
    Regime,
)
from iot_machine_learning.domain.entities.market.prediction.resolver import (
    OutcomeResolver,
)
from iot_machine_learning.domain.entities.market.replay import (
    evaluate_window,
    wf_windows,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.adaptation_repository import (
    AdaptationRepository,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_db_connection import (
    ZeninMarketDbConnection,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_prediction_repository import (
    MarketPredictionRepository,
    row_to_prediction,
)
from sqlalchemy import text

pytestmark = pytest.mark.integration

TS0 = 4_000_000.0
SYMBOL = "TEST-WF"
PREFIX = "TEST-WF-%"


def _candle(timestamp: float) -> Candle:
    return Candle(
        symbol=SYMBOL,
        timestamp=timestamp,
        data_status=DataStatus.REPLAY,
        source_provider="csv_replay",
        venue="NASDAQ",
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.0,
        volume=1000,
        interval_seconds=3600,
        vwap=100.0,
        trade_count=10,
        adjusted=False,
    )


def _prediction(*, ts: float, expert: str, pid: str, prob_up: float = 0.70) -> Prediction:
    return Prediction(
        prediction_id=pid,
        observation=_candle(ts - 3600),
        horizon_seconds=3600,
        timestamp=ts,
        entry_price=100.0,
        expected_return=0.02,
        probability_up=prob_up,
        confidence=0.60,
        interval=PredictionInterval(lower=-0.01, upper=0.05, confidence_level=0.90),
        regime=Regime.BEAR,
        strategy=expert,
    )


class _Prices:
    def __init__(self, closes: dict[float, float]) -> None:
        self.closes = closes

    def last_close(self, at_or_before: float) -> float | None:
        available = [t for t in self.closes if t <= at_or_before]
        if not available:
            return None
        return self.closes[max(available)]


@pytest.fixture(scope="module")
def engine():
    if not ZeninMarketDbConnection.health_check():
        pytest.skip("MySQL zenin_market no disponible")
    from iot_machine_learning.infrastructure.persistence.sql.zenin_market.migrations import (
        apply_migrations,
    )

    apply_migrations()
    with ZeninMarketDbConnection.get_connection() as conn:
        conn.execute(
            text("DELETE FROM market_predictions WHERE prediction_id LIKE :p"),
            {"p": PREFIX},
        )
        conn.execute(text("DELETE FROM adaptation_proposals WHERE proposal_id LIKE 'wf-TEST-WF-%'"))
        conn.execute(text("DELETE FROM model_versions WHERE reason LIKE 'wf TEST-WF%'"))
    return ZeninMarketDbConnection.get_engine()


def _resolve_and_record(
    repo: MarketPredictionRepository,
    predictions: list[Prediction],
    closes: dict[float, float],
) -> None:
    repo.save_batch(predictions)
    resolver = OutcomeResolver()
    prices = _Prices(closes)
    pending = repo.pending_outcomes(symbol=SYMBOL)
    batch = resolver.resolve((row_to_prediction(r) for r in pending), prices)
    repo.save_batch(list(batch.resolved))
    for p in predictions:
        repo.record(p.prediction_id)


class TestWalkForwardCircuit:
    def test_train_adapt_test_evaluate(self, engine) -> None:
        with engine.begin() as conn:
            repo = MarketPredictionRepository(conn)
            adaptation = AdaptationRepository(conn)
            analyzer = PerformanceAnalyzer()

            # ── TRAIN: dos expertos, outcomes reales (bear) ──
            # momentum acierta 20/20 (P=0.7, precio sube; Wilson LB > 0.5
            # con n=20); naive falla 0/20 (P=0.3, precio sube): así el
            # softmax y el guardrail estadístico tienen algo que decidir.
            train_preds = [
                _prediction(
                    ts=TS0 + i * 3600,
                    expert="momentum",
                    pid=f"TEST-WF-momentum-{int(TS0 + i * 3600)}-3600",
                    prob_up=0.70,
                )
                for i in range(20)
            ] + [
                _prediction(
                    ts=TS0 + i * 3600,
                    expert="naive",
                    pid=f"TEST-WF-naive-{int(TS0 + i * 3600)}-3600",
                    prob_up=0.30,
                )
                for i in range(20)
            ]
            closes = {TS0 + i * 3600: 101.0 for i in range(20)}
            closes.update({TS0 + i * 3600: 99.0 for i in range(20, 24)})
            _resolve_and_record(repo, train_preds, closes)

            train_scores = analyzer.analyze(
                repo.expert_performance(symbol=SYMBOL, since=TS0 - 1, until=TS0 + 7 * 3600)
            )
            assert {s.expert for s in train_scores} == {"momentum", "naive"}
            assert all(s.regime == "bear" for s in train_scores)

            # Guard estándar (min_history_days=2): 1 día observado → REJECT.
            proposer = WeightProposer(min_n=3)
            guard = AdaptationGuard(min_n=3, min_history_days=2)
            proposals = proposer.propose(train_scores, {}, parent_version="1")
            assert proposals
            rejected = 0
            for p in proposals:
                score = next(
                    s
                    for s in train_scores
                    if (s.expert, s.regime, s.horizon_seconds)
                    == (p.expert, p.regime, p.horizon_seconds)
                )
                verdict = guard.evaluate(
                    p,
                    history_days=score.history_days,
                    context_weights_after={"momentum": 0.5, "naive": 0.5},
                )
                assert not verdict.passed
                adaptation.record_proposal(
                    p,
                    verdict,
                    proposal_id=f"wf-TEST-WF-W0-{p.expert}-rej",
                )
                rejected += 1
            assert rejected == len(proposals)
            assert adaptation.proposal_history(status="rejected")[0]["proposal_id"].startswith(
                "wf-TEST-WF-"
            )

            # Guard permisivo (1 día): acepta → versión creada.
            guard_ok = AdaptationGuard(min_n=3, min_history_days=1)
            accepted = []
            by_context = {}
            for s in train_scores:
                by_context.setdefault((s.regime, s.horizon_seconds), []).append(s)
            vectors = {}
            for (regime, horizon), group in by_context.items():
                vector, _ = proposer.propose_vector(regime, horizon, group, {})
                vectors[(regime, horizon)] = vector
            for p in proposals:
                verdict = guard_ok.evaluate(
                    p,
                    history_days=2,
                    context_weights_after=vectors[(p.regime, p.horizon_seconds)],
                )
                adaptation.record_proposal(
                    p,
                    verdict,
                    proposal_id=f"wf-TEST-WF-W0-{p.expert}-ok",
                )
                if verdict.passed:
                    accepted.append(p)
            assert accepted
            version_id = adaptation.create_version(
                weights={
                    f"*|bear|{p.horizon_seconds}s": dict(vectors[(p.regime, p.horizon_seconds)])
                    for p in accepted
                },
                calibration={},
                reason="wf TEST-WF W0: propuestas aceptadas",
                proposal_ids=[p.context_label for p in accepted],
            )
            latest = adaptation.latest_version()
            assert int(latest["version_id"]) == version_id
            weights_by_context = {
                str(ctx): dict(w) for ctx, w in json.loads(latest["weights"]).items()
            }

            # ── TEST: mismos expertos, evaluate_window con pesos reales ──
            test_preds = [
                _prediction(
                    ts=TS0 + 8 * 3600 + i * 3600,
                    expert="momentum",
                    pid=f"TEST-WF-momentum-{int(TS0 + (8 + i) * 3600)}-3600",
                    prob_up=0.70,
                )
                for i in range(2)
            ] + [
                _prediction(
                    ts=TS0 + 8 * 3600 + i * 3600,
                    expert="naive",
                    pid=f"TEST-WF-naive-{int(TS0 + (8 + i) * 3600)}-3600",
                    prob_up=0.30,
                )
                for i in range(2)
            ]
            test_closes = {TS0 + 9 * 3600: 101.5, TS0 + 10 * 3600: 101.5}
            _resolve_and_record(repo, test_preds, test_closes)

            test_scores = analyzer.analyze(
                repo.expert_performance(
                    symbol=SYMBOL,
                    since=TS0 + 8 * 3600 - 1,
                    until=TS0 + 12 * 3600,
                )
            )
            evals = evaluate_window(test_scores, weights_by_context, regime="bear")
            assert len(evals) == 1
            assert evals[0].horizon_seconds == 3600
            assert set(evals[0].weights) == {"momentum", "naive"}
            assert evals[0].model.model_accuracy > 0
            # El modelo está ponderado por la versión creada en el TRAIN.
            stored = weights_by_context["*|bear|3600s"]
            assert evals[0].weights == pytest.approx(stored)
            assert evals[0].n == 8  # 2 momentum + 2 naive, todos evaluados

            # El splitter y el régimen sirven de etiqueta para el reporte.
            candles = [_candle(TS0 + i * 3600) for i in range(24 * 24)]
            windows = wf_windows(
                candles,
                train_seconds=14 * 86400,
                test_seconds=7 * 86400,
                step_seconds=7 * 86400,
            )
            assert windows and windows[0].test_start == windows[0].train_end

    def test_cleanup(self, engine) -> None:
        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM market_predictions WHERE prediction_id LIKE :p"),
                {"p": PREFIX},
            )
            conn.execute(
                text("DELETE FROM adaptation_proposals WHERE proposal_id LIKE 'wf-TEST-WF-%'")
            )
            conn.execute(text("DELETE FROM model_versions WHERE reason LIKE 'wf TEST-WF%'"))
            conn.execute(
                text("UPDATE model_versions SET is_active = 1 ORDER BY version_id DESC LIMIT 1")
            )
