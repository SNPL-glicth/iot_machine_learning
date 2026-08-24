"""Integration — circuito FASE 7: persistencia → espera → resolver → historial.

Save -> pending -> OutcomeResolver -> save -> record -> performance history,
contra MySQL zenin_market real (contenedor `mysql`, variables MYSQL_* del
.env). Usa prediction_ids con prefijo TEST- y los elimina al final para no
ensuciar el store.

Requiere MySQL corriendo; si no, se salta (mismo criterio que
test_market_db_circuit.py).
"""

from __future__ import annotations

import pytest
from iot_machine_learning.domain.entities.market import Candle, DataStatus
from iot_machine_learning.domain.entities.market.prediction import (
    Prediction,
    PredictionInterval,
    Regime,
)
from iot_machine_learning.domain.entities.market.prediction.resolver import (
    OutcomeResolver,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_db_connection import (
    ZeninMarketDbConnection,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_prediction_repository import (
    MarketPredictionRepository,
    row_to_prediction,
)

pytestmark = pytest.mark.integration

TS0 = 2_000_000.0
SYMBOL = "TEST"

_PREFIX = "TEST-"


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
        interval_seconds=60,
        vwap=100.0,
        trade_count=10,
        adjusted=False,
    )


def _prediction(*, timestamp: float, horizon: int, prediction_id: str) -> Prediction:
    return Prediction(
        prediction_id=prediction_id,
        observation=_candle(timestamp - 60),
        horizon_seconds=horizon,
        timestamp=timestamp,
        entry_price=100.0,
        expected_return=0.02,
        probability_up=0.70,
        confidence=0.60,
        interval=PredictionInterval(lower=-0.01, upper=0.05, confidence_level=0.90),
        regime=Regime.BULL,
        strategy="test_strategy",
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
    from sqlalchemy import text

    with ZeninMarketDbConnection.get_connection() as conn:
        conn.execute(text("DELETE FROM market_predictions WHERE prediction_id LIKE 'TEST-%'"))
    return ZeninMarketDbConnection.get_engine()


class TestPersistenceCircuit:
    def test_full_circuit(self, engine) -> None:
        from sqlalchemy import text

        with engine.begin() as conn:
            repo = MarketPredictionRepository(conn)

            due = _prediction(
                timestamp=TS0,
                horizon=900,
                prediction_id=f"{_PREFIX}due",
            )
            not_due = _prediction(
                timestamp=TS0,
                horizon=3600,
                prediction_id=f"{_PREFIX}not-due",
            )
            invalidated = _prediction(
                timestamp=TS0,
                horizon=60,
                prediction_id=f"{_PREFIX}invalidated",
            ).invalidate("provider_gap")

            assert repo.save_batch([due, not_due, invalidated]) == 3

            # El upsert es idempotente: re-guardar no duplica.
            assert repo.save_batch([due]) == 1
            rows = conn.execute(
                text(
                    "SELECT COUNT(*) AS n FROM market_predictions "
                    "WHERE prediction_id LIKE 'TEST-%'"
                )
            ).mappings().one()
            assert rows["n"] == 3

            # Fase "espera horizonte": ambos siguen sin resolver; el
            # resolver decide por vencimiento (horizonte 3600 aún no cubre).
            pending = repo.pending_outcomes(symbol=SYMBOL)
            assert {r["prediction_id"] for r in pending} == {
                due.prediction_id,
                not_due.prediction_id,
            }

            resolver = OutcomeResolver()
            prices = _Prices({TS0 + 900: 101.5})
            batch = resolver.resolve(
                (row_to_prediction(r) for r in pending), prices
            )
            assert batch.resolved_count == 1
            assert batch.waiting_count == 1
            assert batch.resolved[0].prediction_id == not_due.prediction_id
            assert batch.still_waiting[0].prediction_id == due.prediction_id
            repo.save_batch(list(batch.resolved))
            repo.save_batch(list(batch.still_waiting))
            assert {r["prediction_id"] for r in repo.pending_outcomes(symbol=SYMBOL)} == {
                due.prediction_id
            }

            # El record: dirección correcta, dentro del intervalo, reward > 0.
            record = repo.record(not_due.prediction_id)
            assert record is not None
            assert record["status"] == "rewarded"
            assert record["direction_correct"] == 1
            assert record["within_interval"] == 1
            assert record["reward_total"] > 0

            # La invalidada jamás recibe reward.
            rec_inv = repo.record(invalidated.prediction_id)
            assert rec_inv["status"] == "invalidated"
            assert rec_inv["invalidation_reason"] == "provider_gap"
            assert rec_inv["reward_total"] is None

            # Historial: la serie agrega por horizonte/estrategia/día.
            history = repo.performance_history(symbol=SYMBOL)
            by_horizon = {r["key"]: r for r in history["by_horizon"]}
            assert by_horizon[3600]["evaluated"] == 1
            assert by_horizon[3600]["hits"] == 1
            assert by_horizon[3600]["reward"] > 0
            assert {r["key"] for r in history["by_strategy"]} == {"test_strategy"}
            assert {r["key"] for r in history["by_regime"]} == {"bull"}

            # Dashboard (FASE 7.5): overall_stats alimenta el overview.
            stats = repo.overall_stats(symbol=SYMBOL)
            assert stats["predictions"] == 3
            assert stats["evaluated"] == 1
            assert stats["pending"] == 1
            assert stats["invalidated"] == 1
            assert stats["hits"] == 1
            assert stats["brier"] is not None and stats["brier"] > 0
            assert stats["reward"] > 0

            # La curva de calibración usa los buckets del historial.
            buckets = {
                r["bucket"]: r for r in history["by_confidence"]
            }
            assert round(buckets[0.7]["avg_probability"], 2) == 0.70
            assert buckets[0.7]["evaluated"] == 1
            assert buckets[0.7]["hits"] == 1

    def test_cleanup(self, engine) -> None:
        from sqlalchemy import text

        with engine.begin() as conn:
            conn.execute(
                text("DELETE FROM market_predictions WHERE prediction_id LIKE 'TEST-%'")
            )
