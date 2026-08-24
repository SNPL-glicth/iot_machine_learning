"""Live Shadow vs Replay parity (FASE 6).

El principio arquitectónico: la diferencia entre Replay y Live está en
la fuente/reloj, no en la lógica de ZENIN. Misma secuencia de eventos:

    * replay: HistoricalFeed + ReplayClock
    * live:   LiveFeed + LiveClock

debe producir predicciones equivalentes (timestamp, features, prediction,
horizon). Y cuando el feed live pierde datos, el shadow lo hace visible:

    * GAP_DETECTED (expected/received) registrado en el feed;
    * predicciones emitidas sobre contexto incompleto → INVALIDATED
      al emitir con ``reason="provider_gap"`` — jamás producen reward.
"""

from __future__ import annotations

import pytest
from iot_machine_learning.domain.entities.market import Candle, ConnectionState, DataStatus
from iot_machine_learning.domain.entities.market.prediction import PredictionStatus
from iot_machine_learning.domain.entities.market.replay import (
    ClockRollbackError,
    LiveClock,
    MarketReplayEngine,
    ReplayClock,
    ReplayEngineConfig,
)
from iot_machine_learning.infrastructure.adapters.market import (
    GapDetected,
    LiveFeed,
    LiveShadowRunner,
)


def _candle(ts: int, close: float, interval: int = 60, *, symbol: str = "NVDA") -> Candle:
    return Candle(
        symbol=symbol,
        timestamp=float(ts),
        data_status=DataStatus.REPLAY,
        source_provider="parity-test",
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=100.0,
        interval_seconds=interval,
    )


def _sequence(n: int, *, close0: float = 100.0, step: float = 0.01) -> tuple[Candle, ...]:
    return tuple(
        _candle(ts, round(close0 + ts * step, 4))
        for ts in range(60, (n + 1) * 60, 60)
    )


class FragmentFeed:
    """Feed congelado (contrato HistoricalFeed) sobre una secuencia dada."""

    def __init__(self, candles: tuple[Candle, ...]) -> None:
        self.symbol = candles[0].symbol
        self.resolution_seconds = 60
        self._candles = candles

    def iter_events(self):
        yield from self._candles


class GappedFeed(FragmentFeed):
    """Feed que omite eventos dentro de [drop_start, drop_end] (pérdida)."""

    def __init__(self, candles: tuple[Candle, ...], drop_start: float, drop_end: float) -> None:
        super().__init__(tuple(c for c in candles if not drop_start <= c.timestamp <= drop_end))
        self.drop_window = (drop_start, drop_end)


_CORE_FIELDS = (
    "timestamp",
    "entry_price",
    "expected_return",
    "probability_up",
    "confidence",
    "horizon_seconds",
    "interval",
)


def _core(pred) -> tuple:
    return tuple(getattr(pred, f) for f in _CORE_FIELDS)


def _cores(preds) -> dict[str, tuple]:
    return {p.prediction_id: _core(p) for p in preds}


def _run_replay(feed: FragmentFeed, *, clock=None, horizons: tuple[int, ...] = (60, 300)):
    engine = MarketReplayEngine(
        ReplayEngineConfig(
            symbol=feed.symbol,
            feed=feed,
            interval_seconds=60,
            horizons_seconds=horizons,
            initial_clock=clock,
        )
    )
    return engine.run()


def _run_shadow(feed: FragmentFeed, *, clock=None, horizons: tuple[int, ...] = (60, 300)):
    live = LiveFeed(
        symbol=feed.symbol,
        historical_feed=feed,
        expected_interval_seconds=60,
    )
    runner = LiveShadowRunner(
        live,
        ReplayEngineConfig(
            symbol=feed.symbol,
            feed=live,
            interval_seconds=60,
            horizons_seconds=horizons,
            initial_clock=clock,
        ),
    )
    return runner.run()


class TestLiveShadowParity:
    def test_gapless_sequence_equivalent_replay_and_live(self) -> None:
        """Misma secuencia sin gaps: replay y live shadow idénticos."""
        fragment = FragmentFeed(_sequence(n=120))
        replay = _run_replay(fragment)
        shadow = _run_shadow(
            fragment,
            clock=LiveClock(now=fragment._candles[0].timestamp),
        )

        assert _cores(shadow.all_predictions) == _cores(replay.predictions)
        assert shadow.gaps == ()
        assert shadow.invalidated_by_gap == ()
        assert shadow.transitions == ()

    def test_live_and_replay_clock_equivalent_for_engine(self) -> None:
        """El engine no distingue el reloj: LiveClock == ReplayClock."""
        fragment = FragmentFeed(_sequence(n=120))
        first_ts = fragment._candles[0].timestamp
        with_live = _run_replay(fragment, clock=LiveClock(now=first_ts))
        with_replay = _run_replay(fragment, clock=ReplayClock(now=first_ts))
        assert _cores(with_live.predictions) == _cores(with_replay.predictions)

    def test_gap_detected_with_expected_and_received(self) -> None:
        """Condición 3: el gap es visible (expected/received), no silencioso."""
        candles = _sequence(n=120)
        gapped = GappedFeed(candles, drop_start=3600.0, drop_end=3600.0)
        live = LiveFeed(
            symbol="NVDA",
            historical_feed=gapped,
            expected_interval_seconds=60,
        )
        list(live.iter_events())
        gaps = live.gaps
        assert len(gaps) == 1
        gap = gaps[0]
        assert isinstance(gap, GapDetected)
        assert gap.expected_timestamp == 3600.0
        assert gap.received_timestamp == 3660.0
        assert gap.gap_seconds == 120.0

    def test_gap_invalidates_at_emission_before_any_reward(self) -> None:
        """Condiciones 3/4: contexto incompleto → INVALIDATED reason=provider_gap
        al emitir; ni siquiera un horizonte corto que vencería dentro del
        fragmento produce reward."""
        candles = _sequence(n=120)
        # Se pierde la vela 60s (ts 3600): llega la 120s (ts 3660).
        # El contexto queda incompleto hasta que la vela recibida cierra
        # (3720): las predicciones emitidas en 3660 y 3720 se invalidan.
        gapped = GappedFeed(candles, drop_start=3600.0, drop_end=3600.0)
        shadow = _run_shadow(
            gapped,
            clock=LiveClock(now=candles[0].timestamp),
            horizons=(60, 300, 900, 3600),
        )

        assert len(shadow.gaps) == 1
        assert shadow.degraded_windows == (type(shadow.degraded_windows[0])(3600.0, 3720.0),)
        affected = shadow.invalidated_by_gap
        assert affected, "deben existir predicciones invalidadas por gap"
        for pred in affected:
            assert pred.status == PredictionStatus.INVALIDATED
            assert pred.invalidation_reason == "provider_gap"
            assert pred.outcome is None
            assert pred.reward is None
            assert pred.can_produce_reward is False
            assert pred.timestamp in (3660.0, 3720.0)

    def test_gap_affected_prediction_still_matches_replay(self) -> None:
        """La lógica es la misma: el gap NO cambia features/predicción,
        solo la honestidad (invalidate al emitir en vez de reward)."""
        candles = _sequence(n=120)
        gapped = GappedFeed(candles, drop_start=3600.0, drop_end=3600.0)

        replay = _run_replay(gapped)
        shadow = _run_shadow(
            gapped,
            clock=LiveClock(now=candles[0].timestamp),
        )

        replay_map = {p.prediction_id: p for p in replay.predictions}
        shadow_map = _cores(shadow.all_predictions)
        assert set(shadow_map) == set(replay_map)
        for pred_id, core in shadow_map.items():
            assert core == _core(replay_map[pred_id])

        for pred in shadow.invalidated_by_gap:
            counterpart = replay_map[pred.prediction_id]
            assert counterpart.outcome is not None, (
                "el replay sí resolvió la contraparte: solo el shadow "
                "invalida el contexto incompleto"
            )
            assert counterpart.status == PredictionStatus.REWARDED

    def test_healthy_predictions_untouched_by_gap(self) -> None:
        """Solo el contexto incompleto se invalida; el resto sigue normal."""
        candles = _sequence(n=140)
        gapped = GappedFeed(candles, drop_start=3600.0, drop_end=3600.0)
        shadow = _run_shadow(
            gapped,
            clock=LiveClock(now=candles[0].timestamp),
        )
        affected = {p.prediction_id for p in shadow.invalidated_by_gap}
        feed_end = {
            p.prediction_id
            for p in shadow.invalidated
            if p.invalidation_reason != "provider_gap"
        }
        healthy = [
            p
            for p in shadow.predictions
            if p.prediction_id not in affected | feed_end
        ]
        assert healthy
        assert all(p.status != PredictionStatus.INVALIDATED for p in healthy)
        assert all(p.reward is not None for p in healthy if p.outcome is not None)
        assert all(
            p.status == PredictionStatus.INVALIDATED
            and p.invalidation_reason == "feed_ended"
            for p in shadow.invalidated
            if p.prediction_id in feed_end
        )

    def test_recovered_state_transition_visible(self) -> None:
        """Condición 4: DEGRADED → RECOVERED → CONNECTED queda registrado."""
        candles = _sequence(n=120)
        gapped = GappedFeed(candles, drop_start=3600.0, drop_end=3600.0)
        shadow = _run_shadow(
            gapped,
            clock=LiveClock(now=candles[0].timestamp),
        )

        states = [t.state for t in shadow.transitions]
        assert states[:2] == [ConnectionState.DEGRADED, ConnectionState.RECOVERED]
        assert states[2] == ConnectionState.CONNECTED
        assert shadow.transitions[-1].state == ConnectionState.CONNECTED
        assert shadow.transitions[-1].at_timestamp == 3720.0

    def test_live_feed_emits_same_observation_objects(self) -> None:
        """Condición 1: LiveFeed entrega el mismo MarketObservation sin
        alterarlo (mismo tipo, mismos campos, sin retagging)."""
        candles = _sequence(n=10)
        live = LiveFeed(
            symbol="NVDA",
            historical_feed=FragmentFeed(candles),
            expected_interval_seconds=60,
        )
        emitted = list(live.iter_events())
        assert len(emitted) == len(candles)
        for original, delivered in zip(candles, emitted, strict=True):
            assert type(delivered) is type(original)
            assert delivered.timestamp == original.timestamp
            assert delivered.close == original.close
            assert delivered.data_status is original.data_status

    def test_manual_state_update_records_transition(self) -> None:
        """update_state externo (ej. websocket caído) queda en el historial."""
        live = LiveFeed(
            symbol="NVDA",
            historical_feed=FragmentFeed(_sequence(n=5)),
            expected_interval_seconds=60,
        )
        live.update_state(ConnectionState.DISCONNECTED, at_timestamp=123.0)
        live.update_state(ConnectionState.RECONNECTING, at_timestamp=130.0)
        live.update_state(ConnectionState.RECOVERED, at_timestamp=140.0)
        states = [t.state for t in live.transitions]
        assert states == [
            ConnectionState.DISCONNECTED,
            ConnectionState.RECONNECTING,
            ConnectionState.RECOVERED,
        ]

    def test_live_clock_rollback_rejected(self) -> None:
        """LiveClock mantiene el invariante del reloj: nada de retrocesos."""
        clock = LiveClock(now=100.0)
        with pytest.raises(ClockRollbackError):
            clock.advance_to(99.0)

    def test_config_rejects_invalid_degraded_windows(self) -> None:
        with pytest.raises(ValueError):
            ReplayEngineConfig(
                symbol="NVDA",
                feed=FragmentFeed(_sequence(n=10)),
                interval_seconds=60,
                horizons_seconds=(60,),
                degraded_windows=((3660.0, 3600.0),),
            )

    def test_replay_default_never_invalidates_provider_gap(self) -> None:
        """Condición 1: el replay clásico (sin ventanas degradadas) es
        idéntico aunque la secuencia tenga huecos: sin reason=provider_gap."""
        candles = _sequence(n=120)
        gapped = GappedFeed(candles, drop_start=3600.0, drop_end=3600.0)
        replay = _run_replay(gapped)
        assert all(
            p.invalidation_reason != "provider_gap" for p in replay.predictions
        )
        assert all(
            p.invalidation_reason != "provider_gap" for p in replay.invalidated
        )