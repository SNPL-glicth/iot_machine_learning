"""MarketReplayEngine (FASE 5 → FASE 6) — reproduce el pasado sin mirar el futuro.

Regla de oro: el reloj del replay es la única fuente de verdad. Cuando
el reloj está en ``T``, nada con timestamp mayor a ``T`` es observable:
una vela entra a la ventana de features recién cuando el reloj la cierra
(``ts_close <= now``), y un Outcome se resuelve recién cuando el reloj
alcanza el vencimiento, usando solo el último cierre conocido.

Ciclo por evento (feed ordenado no-decreciente):
    1. avanzar el reloj al timestamp del evento (monótono, validado);
    2. cerrar velas en formación cuyo ``ts_close <= now`` -> ventana;
    3. resolver outcomes vencidos (``obs_ts + horizon <= now``);
    4. si el cierre alimentó la ventana, predecir cada horizonte
       (determinista: mismas velas cerradas -> misma predicción);
    5. registrar la vela del evento como "en formación".

El mismo engine sirve a replay y a futuro live: solo cambia el feed y
el reloj (lógico vs real); el contrato de no ver el futuro es idéntico.

FASE 6: Usa Protocol ``Clock`` en lugar de ``ReplayClock`` específico.
FASE 6: ``degraded_windows`` (modo shadow) — predicciones emitidas en
contexto incompleto se invalidan al emitir (reason=provider_gap) y jamás
producen reward. Vacío por defecto: el replay clásico no cambia.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from .. import DataStatus, MarketObservation
from ..observations import Candle
from ..prediction.evaluation import evaluate_prediction
from ..prediction.lifecycle import PredictionStatus
from ..prediction.outcome import Outcome
from ..prediction.prediction import Prediction
from ..prediction.reward import RewardConfig, compute_reward
from .feature_window import FeatureWindow
from .feed import HistoricalFeed
from .clock import Clock, ReplayClock
from .config import ReplayEngineConfig
from .result import ReplayRunResult
from .emitter import emit_predictions


class MarketReplayEngine:
    """Motor walk-forward: anticipa solo lo ya ocurrido."""

    def __init__(self, config: ReplayEngineConfig) -> None:
        self.config = config

    def run(self) -> ReplayRunResult:
        cfg = self.config

        self._emitted_order: list[Prediction] = []
        self._invalidated_order: list[Prediction] = []
        self._latency_samples: list[int] = []
        self._emit_count: int = 0
        clock: Clock | None = None
        window = FeatureWindow(symbol=cfg.symbol)
        open_candles: dict[float, Candle] = {}
        pending: dict[int, list[Prediction]] = {}
        resolved_by_id: dict[str, Prediction] = {}
        invalidated: list[Prediction] = []

        for event in cfg.feed.iter_events():
            if not isinstance(event, MarketObservation):
                raise TypeError(
                    f"feed entregó {type(event).__name__}, "
                    "se espera MarketObservation"
                )
            if event.symbol != cfg.symbol:
                raise ValueError(
                    f"feed fuera de símbolo: esperaba {cfg.symbol}, "
                    f"obtuvo {event.symbol!r}"
                )

            if clock is None:
                clock = cfg.initial_clock or ReplayClock(now=event.timestamp)
            else:
                # Monotonía validada por el reloj: retroceso = error.
                clock = clock.advance_to(event.timestamp)

            # 1) cerrar velas en formación ya vencidas bajo este reloj.
            matured = sorted(
                ts
                for ts, candle in open_candles.items()
                if candle.timestamp + candle.interval_seconds <= clock.now
            )
            if matured:
                for ts in matured:
                    candle = open_candles.pop(ts)
                    window = window.append_closed(candle)
                window = FeatureWindow(
                    symbol=cfg.symbol,
                    candles=window.candles[-cfg.feature_window_size :],
                )

            # 2) resolver outcomes vencidos con el último cierre conocido.
            due = sorted(ts for ts in pending if ts <= clock.now)
            for resolve_ts in due:
                for pred in pending.pop(resolve_ts):
                    final = window.last_close()
                    if final is None:
                        invalidated.append(pred)
                        continue
                    outcome = Outcome.from_prices(
                        symbol=cfg.symbol,
                        ref_timestamp=pred.observation.timestamp,
                        ref_price=pred.entry_price,
                        horizon_seconds=pred.horizon_seconds,
                        final_price=final,
                        measured_at=clock.now,
                    )
                    evaluation = evaluate_prediction(pred, outcome)
                    reward = compute_reward(
                        pred, outcome, evaluation, cfg.reward_config
                    )
                    resolved = dataclasses.replace(
                        pred,
                        status=PredictionStatus.REWARDED,
                        outcome=outcome,
                        evaluation=evaluation,
                        reward=reward,
                    )
                    resolved_by_id[resolved.prediction_id] = resolved

            # 3) predecir si este cierre alimentó la ventana.
            if matured:
                observation = window.last_closed()
                if (
                    observation is not None
                    and window.size >= cfg.predictor_lookback + 1
                ):
                    self._emit_count = emit_predictions(
                        cfg=cfg,
                        observation=observation,
                        window=window,
                        clock=clock,
                        pending=pending,
                        resolved_by_id=resolved_by_id,
                        emitted_order=self._emitted_order,
                        invalidated_order=self._invalidated_order,
                        latency_samples=self._latency_samples,
                        emit_count=self._emit_count,
                        latency_sample_every=cfg.latency_sample_every,
                    )

            # 4) la vela del evento pasa a "en formación".
            if isinstance(event, Candle):
                if event.interval_seconds != cfg.interval_seconds:
                    raise ValueError(
                        f"vela {event.timestamp!r} con intervalo "
                        f"{event.interval_seconds}s, feed espera {cfg.interval_seconds}s"
                    )
                open_candles[event.timestamp] = event

        # Feed agotado: lo que no venció jamás se resuelve → INVALIDATED
        # terminal (FASE 7: el store debe reflejar el contrato del ciclo
        # de vida, no dejar predicciones colgadas en PENDING).
        feed_end_invalidated = [
            p.invalidate("feed_ended") for bucket in pending.values() for p in bucket
        ]
        # FASE 6: primero las invalidadas al emitir (contexto degradado),
        # luego las que el feed dejó sin resolver.
        invalidated = self._invalidated_order + feed_end_invalidated

        final_by_id = {
            p.prediction_id: p for p in feed_end_invalidated
        }
        final_by_id.update(resolved_by_id)
        predictions = tuple(
            final_by_id.get(p.prediction_id, p) for p in self._emitted_order
        )
        return ReplayRunResult(
            symbol=cfg.symbol,
            predictions=predictions,
            outcomes=tuple(
                p.outcome for p in predictions if p.outcome is not None
            ),
            invalidated=tuple(invalidated),
            latency_ns=tuple(self._latency_samples),
        )