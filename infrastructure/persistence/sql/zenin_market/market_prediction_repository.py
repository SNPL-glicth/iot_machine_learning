"""MarketPredictionRepository (FASE 7) — persistencia del loop live.

Persiste el ciclo completo en una sola tabla (``market_predictions``):
snapshot de la observación (JSON), predicción, y los campos del desenlace
(Outcome -> Evaluation -> Reward) hasta que se resuelven.

Regla de FASE 7: append-only. Este repo SOLO guarda y consulta; no existe
ningún camino de escritura hacia el modelo de predicción.

Las funciones de mapeo fila <-> entidad son puras (sin conexión) para que
los tests unitarios las cubran sin MySQL; la clase usa una conexión
SQLAlchemy solo para las operaciones.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from decimal import Decimal
from typing import Any, cast

from sqlalchemy import text
from sqlalchemy.engine import Connection

from iot_machine_learning.domain.entities.market import (
    Candle,
    DataStatus,
    MarketObservation,
    OrderBookSnapshot,
    Quote,
    Trade,
)
from iot_machine_learning.domain.entities.market.prediction import (
    Evaluation,
    InputContext,
    Outcome,
    Prediction,
    PredictionInterval,
    PredictionStatus,
    Regime,
    Reward,
)

__all__ = [
    "MarketPredictionRepository",
    "observation_to_json",
    "observation_from_json",
    "prediction_to_row",
    "row_to_prediction",
]

_OBSERVATION_TYPES = {
    "Candle": Candle,
    "Trade": Trade,
    "Quote": Quote,
    "OrderBookSnapshot": OrderBookSnapshot,
}


def _clean(row: dict[str, Any]) -> dict[str, Any]:
    """MySQL devuelve agregados como Decimal: normaliza a float/int."""
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, Decimal):
            value = float(value)
            if value.is_integer():
                value = int(value)
        cleaned[key] = value
    return cleaned


# ─── Mapeo puro: entidad <-> fila ────────────────────────────────────────


def observation_to_json(observation: MarketObservation) -> str:
    """Serializa una MarketObservation a JSON (snapshot del dominio)."""
    if not isinstance(observation, MarketObservation):
        raise TypeError(
            f"se espera MarketObservation, obtenido {type(observation).__name__}"
        )
    payload: dict[str, Any] = {
        "_type": type(observation).__name__,
        "symbol": observation.symbol,
        "timestamp": observation.timestamp,
        "data_status": observation.data_status.value,
        "source_provider": observation.source_provider,
        "venue": observation.venue,
    }
    if isinstance(observation, Trade):
        payload.update(
            {
                "price": observation.price,
                "size": observation.size,
                "trade_id": observation.trade_id,
                "taker_side": observation.taker_side,
                "conditions": list(observation.conditions),
                "tape": observation.tape,
                "corrected": observation.corrected,
            }
        )
    elif isinstance(observation, Quote):
        payload.update(
            {
                "bid": observation.bid,
                "bid_size": observation.bid_size,
                "ask": observation.ask,
                "ask_size": observation.ask_size,
                "bid_exchange": observation.bid_exchange,
                "ask_exchange": observation.ask_exchange,
                "conditions": list(observation.conditions),
                "tape": observation.tape,
            }
        )
    elif isinstance(observation, Candle):
        payload.update(
            {
                "open": observation.open,
                "high": observation.high,
                "low": observation.low,
                "close": observation.close,
                "volume": observation.volume,
                "interval_seconds": observation.interval_seconds,
                "vwap": observation.vwap,
                "trade_count": observation.trade_count,
                "adjusted": observation.adjusted,
            }
        )
    elif isinstance(observation, OrderBookSnapshot):
        payload.update(
            {
                "bids": [list(level) for level in observation.bids],
                "asks": [list(level) for level in observation.asks],
                "reset": observation.reset,
            }
        )
    else:
        raise TypeError(
            f"tipo de observación sin serializador: {type(observation).__name__}"
        )
    return json.dumps(payload, separators=(",", ":"))


def observation_from_json(payload: str) -> MarketObservation:
    """Reconstruye la MarketObservation desde su snapshot JSON."""
    data = json.loads(payload)
    obs_type = data.pop("_type")
    if obs_type not in _OBSERVATION_TYPES:
        raise ValueError(f"tipo de observación desconocido en snapshot: {obs_type!r}")
    data["data_status"] = DataStatus(data["data_status"])
    if "conditions" in data:
        data["conditions"] = tuple(data["conditions"])
    for key in ("bids", "asks"):
        if key in data:
            data[key] = tuple(tuple(level) for level in data[key])
    return cast(MarketObservation, _OBSERVATION_TYPES[obs_type](**data))


def prediction_to_row(prediction: Prediction) -> dict[str, Any]:
    """Serializa una Prediction (con su ciclo de vida) a fila SQL."""
    interval = prediction.interval
    context = prediction.input_context
    outcome = prediction.outcome
    evaluation = prediction.evaluation
    reward = prediction.reward
    return {
        "prediction_id": prediction.prediction_id,
        "symbol": prediction.observation.symbol,
        "horizon_seconds": prediction.horizon_seconds,
        "emitted_at": prediction.timestamp,
        "observation_timestamp": prediction.observation.timestamp,
        "entry_price": prediction.entry_price,
        "expected_return": prediction.expected_return,
        "probability_up": prediction.probability_up,
        "confidence": prediction.confidence,
        "interval_lower": interval.lower if interval else None,
        "interval_upper": interval.upper if interval else None,
        "interval_confidence": (
            interval.confidence_level if interval else None
        ),
        "regime": prediction.regime.value if prediction.regime else None,
        "strategy": prediction.strategy,
        "data_status": (
            cast(DataStatus, context.data_status).value
            if context and context.data_status is not None
            else None
        ),
        "feature_count": context.feature_count if context else None,
        "feature_version": context.feature_version if context else None,
        "observation": observation_to_json(prediction.observation),
        "status": prediction.status.value,
        "invalidation_reason": prediction.invalidation_reason,
        "outcome_measured_at": outcome.measured_at if outcome else None,
        "outcome_final_price": outcome.final_price if outcome else None,
        "outcome_return_realized": outcome.return_realized if outcome else None,
        "direction_correct": evaluation.direction_correct if evaluation else None,
        "magnitude_error": evaluation.magnitude_error if evaluation else None,
        "within_interval": evaluation.within_interval if evaluation else None,
        "calibration_error": evaluation.calibration_error if evaluation else None,
        "reward_direction": reward.direction_component if reward else None,
        "reward_magnitude": reward.magnitude_component if reward else None,
        "reward_calibration": reward.calibration_component if reward else None,
        "reward_execution_costs": reward.execution_costs if reward else None,
        "reward_total": reward.total if reward else None,
    }


def row_to_prediction(row: dict[str, Any]) -> Prediction:
    """Reconstruye una Prediction (con su ciclo de vida) desde una fila."""
    prediction_id = row["prediction_id"]
    observation = observation_from_json(row["observation"])
    interval = None
    if row.get("interval_lower") is not None:
        interval = PredictionInterval(
            lower=row["interval_lower"],
            upper=row["interval_upper"],
            confidence_level=row["interval_confidence"] or 0.90,
        )
    context = None
    if (
        row.get("data_status") is not None
        or row.get("feature_count") is not None
        or row.get("feature_version") is not None
    ):
        context = InputContext(
            data_status=DataStatus(row["data_status"]) if row.get("data_status") else None,
            feature_count=row.get("feature_count"),
            feature_version=row.get("feature_version"),
        )
    outcome = None
    if row.get("outcome_final_price") is not None:
        outcome = Outcome(
            symbol=row["symbol"],
            observation_timestamp=row["observation_timestamp"],
            horizon_seconds=row["horizon_seconds"],
            measured_at=row["outcome_measured_at"],
            final_price=row["outcome_final_price"],
            return_realized=row["outcome_return_realized"],
        )
    evaluation = None
    if row.get("direction_correct") is not None:
        evaluation = Evaluation(
            direction_correct=bool(row["direction_correct"]),
            magnitude_error=row["magnitude_error"],
            within_interval=bool(row["within_interval"]),
            calibration_error=row["calibration_error"],
        )
    reward = None
    if row.get("reward_total") is not None:
        reward = Reward(
            direction_component=row["reward_direction"],
            magnitude_component=row["reward_magnitude"],
            calibration_component=row["reward_calibration"],
            execution_costs=row["reward_execution_costs"],
            total=row["reward_total"],
        )

    return Prediction(
        prediction_id=prediction_id,
        observation=observation,
        horizon_seconds=row["horizon_seconds"],
        timestamp=row["emitted_at"],
        entry_price=row["entry_price"],
        expected_return=row["expected_return"],
        probability_up=row["probability_up"],
        confidence=row["confidence"],
        interval=interval,
        regime=Regime(row["regime"]) if row.get("regime") else None,
        strategy=row.get("strategy"),
        input_context=context,
        status=PredictionStatus(row["status"]),
        outcome=outcome,
        evaluation=evaluation,
        reward=reward,
        invalidation_reason=row.get("invalidation_reason"),
    )


# ─── Repositorio (conexión SQLAlchemy) ───────────────────────────────────


class MarketPredictionRepository:
    """Persistencia append-only del ciclo Prediction -> Reward."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    def save_prediction(self, prediction: Prediction) -> None:
        """Upsert de una predicción (idempotente: re-runs no duplican)."""
        self._upsert(prediction_to_row(prediction))

    def save_batch(self, predictions: Iterable[Prediction]) -> int:
        """Upsert de un lote; retorna el número de filas escritas."""
        rows = [prediction_to_row(pred) for pred in predictions]
        if not rows:
            return 0
        for row in rows:
            self._upsert(row)
        return len(rows)

    def pending_outcomes(
        self,
        *,
        symbol: str | None = None,
        before_ts: float | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Predicciones aún sin resolver (PENDING/ACTIVE/WAITING_OUTCOME)."""
        sql = (
            "SELECT * FROM market_predictions "
            "WHERE status IN ('pending', 'active', 'waiting_outcome')"
        )
        params: dict[str, Any] = {}
        if symbol is not None:
            sql += " AND symbol = :symbol"
            params["symbol"] = symbol
        if before_ts is not None:
            sql += " AND emitted_at <= :before_ts"
            params["before_ts"] = before_ts
        sql += " ORDER BY emitted_at"
        rows = self._conn.execute(text(sql), params).mappings().all()
        return tuple(_clean(dict(row)) for row in rows)

    def record(self, prediction_id: str) -> dict[str, Any] | None:
        """Fila completa de una predicción (para el record/auditoría)."""
        row = self._conn.execute(
            text(
                "SELECT * FROM market_predictions "
                "WHERE prediction_id = :prediction_id"
            ),
            {"prediction_id": prediction_id},
        ).mappings().first()
        return dict(row) if row else None

    def recent_records(
        self,
        *,
        symbol: str | None = None,
        status: str | None = None,
        limit: int = 5,
    ) -> tuple[dict[str, Any], ...]:
        """Últimas filas por timestamp de observación (para el record)."""
        sql = "SELECT * FROM market_predictions"
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if symbol is not None:
            clauses.append("symbol = :symbol")
            params["symbol"] = symbol
        if status is not None:
            clauses.append("status = :status")
            params["status"] = status
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY observation_timestamp DESC LIMIT :limit"
        params["limit"] = limit
        rows = self._conn.execute(text(sql), params).mappings().all()
        return tuple(_clean(dict(row)) for row in rows)

    def prediction_records(
        self,
        *,
        symbol: str | None = None,
        since: float | None = None,
        until: float | None = None,
        status: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Filas crudas de predicción (FASE 10 — Prediction Observatory).

        Devuelve el registro completo de la memoria observable de ZENIN,
        ordenado por emisión (cronológico), para el dashboard: conteos por
        estado, agregados por dimensión, calibración, learning curve y
        evidencia. Sin filtros de calidad: el observatorio ve TODO y
        reporta el staling por separado.
        """
        where, params = self._filters(symbol, since, until)
        if status is not None:
            where += " AND status = :status"
            params["status"] = status
        rows = self._conn.execute(
            text(
                f"""
                SELECT *
                FROM market_predictions
                {where}
                ORDER BY emitted_at ASC
                """
            ),
            params,
        ).mappings().all()
        return tuple(_clean(dict(row)) for row in rows)

    def performance_history(
        self,
        *,
        symbol: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> dict[str, tuple[dict[str, Any], ...]]:
        """Historial agregado para el tablero de ZENIN.

        Grupos: por horizonte, por estrategia, por régimen, por bucket de
        confianza (¿el 70% declarado es realmente ~70%?) y por día.
        Solo filas evaluadas (status = 'rewarded') alimentan las tasas.
        """
        where, params = self._filters(symbol, since, until)
        return {
            "by_horizon": self._group_by(
                "horizon_seconds",
                where,
                params,
                order="horizon_seconds",
            ),
            "by_strategy": self._group_by("strategy", where, params, order="strategy"),
            "by_regime": self._group_by("regime", where, params, order="regime"),
            "by_confidence": self._confidence_buckets(where, params),
            "by_day": self._by_day(where, params),
        }

    def overall_stats(
        self,
        *,
        symbol: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> dict[str, Any]:
        """Resumen del tablero: conteos por estado + Brier + reward total."""
        where, params = self._filters(symbol, since, until)
        row = self._conn.execute(
            text(
                f"""
                SELECT
                    COUNT(*) AS predictions,
                    SUM(CASE WHEN status = 'rewarded' THEN 1 ELSE 0 END) AS evaluated,
                    SUM(CASE WHEN status IN ('pending', 'active', 'waiting_outcome')
                             THEN 1 ELSE 0 END) AS pending,
                    SUM(CASE WHEN status = 'invalidated' THEN 1 ELSE 0 END) AS invalidated,
                    SUM(CASE WHEN status = 'archived' THEN 1 ELSE 0 END) AS archived,
                    SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) AS hits,
                    AVG(CASE WHEN status = 'rewarded' THEN
                        POW(probability_up - IF(direction_correct, 1.0, 0.0), 2)
                    END) AS brier,
                    AVG(CASE WHEN status = 'rewarded' THEN magnitude_error END)
                        AS magnitude_error,
                    COALESCE(SUM(reward_total), 0.0) AS reward
                FROM market_predictions
                {where}
                """
            ),
            params,
        ).mappings().one()
        return _clean(dict(row))

    def expert_performance(
        self,
        *,
        symbol: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Rendimiento por (experto, régimen, horizonte) — FASE 8.

        Solo filas evaluadas (status=rewarded, outcome real): el análisis
        de adaptación jamás ve INVALIDATED ni PENDING. Incluye días
        distintos observados por contexto (guardrail de historial).
        """
        where, params = self._filters(symbol, since, until)
        rows = self._conn.execute(
            text(
                f"""
                SELECT
                    strategy,
                    regime,
                    horizon_seconds,
                    COUNT(DISTINCT DATE(FROM_UNIXTIME(emitted_at))) AS days,
                    COUNT(*) AS evaluated,
                    SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) AS hits,
                    AVG(calibration_error) AS calibration,
                    AVG(expected_return) AS expected_return,
                    AVG(outcome_return_realized) AS realized_return,
                    AVG(reward_execution_costs) AS execution_costs,
                    STDDEV(
                        CASE WHEN direction_correct = 1
                             THEN ABS(outcome_return_realized)
                             ELSE -ABS(outcome_return_realized) END
                    ) AS risk_std,
                    COALESCE(SUM(reward_total), 0.0) AS reward
                FROM market_predictions
                {where}
                AND status = 'rewarded'
                AND (data_status IS NULL OR data_status <> 'stale')
                GROUP BY strategy, regime, horizon_seconds
                ORDER BY strategy, regime, horizon_seconds
                """
            ),
            params,
        ).mappings().all()
        return tuple(_clean(dict(row)) for row in rows)

    # ─── helpers SQL ──────────────────────────────────────────────────────

    def _upsert(self, row: dict[str, Any]) -> None:
        columns = list(row)
        placeholders = ", ".join(f":{col}" for col in columns)
        updates = ", ".join(
            f"{col} = VALUES({col})" for col in columns if col != "prediction_id"
        )
        self._conn.execute(
            text(
                f"INSERT INTO market_predictions ({', '.join(columns)}) "
                f"VALUES ({placeholders}) "
                f"ON DUPLICATE KEY UPDATE {updates}"
            ),
            row,
        )

    @staticmethod
    def _filters(
        symbol: str | None,
        since: float | None,
        until: float | None,
    ) -> tuple[str, dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if symbol is not None:
            clauses.append("symbol = :symbol")
            params["symbol"] = symbol
        if since is not None:
            clauses.append("emitted_at >= :since")
            params["since"] = since
        if until is not None:
            clauses.append("emitted_at <= :until")
            params["until"] = until
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def _group_by(
        self,
        column: str,
        where: str,
        params: dict[str, Any],
        *,
        order: str,
    ) -> tuple[dict[str, Any], ...]:
        rows = self._conn.execute(
            text(
                f"""
                SELECT {column} AS `key`,
                       COUNT(*) AS predictions,
                       SUM(CASE WHEN status = 'rewarded' THEN 1 ELSE 0 END) AS evaluated,
                       SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) AS hits,
                       AVG(calibration_error) AS calibration,
                       AVG(magnitude_error) AS magnitude_error,
                       COALESCE(SUM(reward_total), 0.0) AS reward
                FROM market_predictions
                {where}
                GROUP BY {column}
                ORDER BY {order}
                """
            ),
            params,
        ).mappings().all()
        return tuple(_clean(dict(row)) for row in rows)

    def _confidence_buckets(
        self,
        where: str,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        rows = self._conn.execute(
            text(
                f"""
                SELECT
                    ROUND(probability_up * 10) / 10 AS bucket,
                    COUNT(*) AS predictions,
                    SUM(CASE WHEN status = 'rewarded' THEN 1 ELSE 0 END) AS evaluated,
                    SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) AS hits,
                    AVG(probability_up) AS avg_probability,
                    AVG(calibration_error) AS calibration
                FROM market_predictions
                {where}
                GROUP BY ROUND(probability_up * 10) / 10
                ORDER BY bucket
                """
            ),
            params,
        ).mappings().all()
        return tuple(_clean(dict(row)) for row in rows)

    def _by_day(
        self,
        where: str,
        params: dict[str, Any],
    ) -> tuple[dict[str, Any], ...]:
        rows = self._conn.execute(
            text(
                f"""
                SELECT
                    DATE(FROM_UNIXTIME(emitted_at)) AS day,
                    COUNT(*) AS predictions,
                    SUM(CASE WHEN status = 'rewarded' THEN 1 ELSE 0 END) AS evaluated,
                    SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) AS hits,
                    COALESCE(SUM(reward_total), 0.0) AS reward
                FROM market_predictions
                {where}
                GROUP BY DATE(FROM_UNIXTIME(emitted_at))
                ORDER BY day
                """
            ),
            params,
        ).mappings().all()
        return tuple(_clean(dict(row)) for row in rows)
