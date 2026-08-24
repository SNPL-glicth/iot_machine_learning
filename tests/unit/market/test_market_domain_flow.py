"""Integración de dominio (sin APIs): Market -> Prediction -> Outcome -> Evaluation -> Reward.

Flujos extremo a extremo con datos sintéticos para verificar que
Fases 0–3 conviven como un sistema, no como colecciones de tests.

Escenario feliz: NVDA a 100.00, predicción +1% @15m (P(up)=0.70),
cierra en 101.20 (+1.2%) -> direction correct, within_interval,
reward positivo.

Escenario adverso: provider desconectado -> INVALIDATED -> reward = None.
"""

from __future__ import annotations

import pytest
from iot_machine_learning.domain.entities.market import DataStatus, Quote
from iot_machine_learning.domain.entities.market.prediction import (
    InvalidTransitionError,
    Outcome,
    Prediction,
    PredictionInterval,
    PredictionStatus,
    RewardConfig,
)

TS = 1_600_000_000.0


def _nvda_quote(**overrides) -> Quote:
    kwargs = dict(
        symbol="NVDA",
        timestamp=TS,
        data_status=DataStatus.REALTIME,
        source_provider="alpaca",
        bid=99.95,
        bid_size=400.0,
        ask=100.05,
        ask_size=350.0,
    )
    kwargs.update(overrides)
    return Quote(**kwargs)


def _prediction(quote: Quote, **overrides) -> Prediction:
    kwargs = dict(
        prediction_id="nvda-15m",
        observation=quote,
        horizon_seconds=900,
        timestamp=quote.timestamp,
        entry_price=quote.midpoint,
        expected_return=0.01,
        probability_up=0.70,
        confidence=0.80,
        interval=PredictionInterval(lower=-0.005, upper=0.025),
    )
    kwargs.update(overrides)
    return Prediction(**kwargs)


def _outcome(prediction: Prediction, final_price: float) -> Outcome:
    return Outcome.from_prices(
        symbol=prediction.observation.symbol,
        ref_timestamp=prediction.observation.timestamp,
        ref_price=prediction.entry_price,
        horizon_seconds=prediction.horizon_seconds,
        final_price=final_price,
    )


class TestHappyPath:
    """NVDA 100.00 -> +1% @15m -> 101.20 -> evaluation -> reward."""

    def test_full_flow_produces_positive_reward(self):
        quote = _nvda_quote()  # midpoint = 100.00
        assert quote.midpoint == pytest.approx(100.00)

        prediction = _prediction(quote)
        outcome = _outcome(prediction, final_price=101.20)

        evaluated = (
            prediction.activate()
            .to_waiting_outcome(outcome)
            .evaluate(outcome)
        )
        assert outcome.return_realized == pytest.approx(0.012)

        evaluation = evaluated.evaluation
        assert evaluation.direction_correct is True
        assert evaluation.within_interval is True
        assert prediction.interval.contains(outcome.return_realized)

        rewarded = evaluated.issue_reward(RewardConfig())
        assert rewarded.status is PredictionStatus.REWARDED
        assert rewarded.reward is not None
        assert rewarded.reward.total > 0
        assert rewarded.reward.direction_component > 0

    def test_persistence_shape_after_flow(self):
        """El reward queda en la entidad final (para persistir en Fase 5)."""
        quote = _nvda_quote()
        prediction = _prediction(quote)
        outcome = _outcome(prediction, final_price=101.20)
        rewarded = (
            prediction.activate()
            .to_waiting_outcome(outcome)
            .evaluate(outcome)
            .issue_reward(RewardConfig())
        )
        assert rewarded.outcome is outcome
        assert rewarded.evaluation is not None
        assert rewarded.reward is not None


class TestAdversarialPath:
    """Provider desconectado -> INVALIDATED -> jamás reward."""

    def test_stale_provider_invalidates_prediction(self):
        # Con el feed degradado (stale), la política de dominio invalida:
        # la predicción jamás llega a reward (Fase 4 conecta el adaptador).
        _nvda_quote(data_status=DataStatus.STALE)
        quote = _nvda_quote()  # feed OK
        prediction = _prediction(quote)
        # El conmutador de data_status marca la predicción como inservible
        # (en Fase 4 esto lo decide el adapter/providers, aquí la política)
        invalidated = prediction.activate().invalidate()
        assert invalidated.status is PredictionStatus.INVALIDATED
        assert invalidated.reward is None
        assert invalidated.outcome is None
        assert invalidated.evaluation is None

    def test_invalidated_cannot_receive_reward(self):
        quote = _nvda_quote()
        prediction = _prediction(quote)
        invalidated = prediction.activate().invalidate()
        with pytest.raises(InvalidTransitionError):
            invalidated.issue_reward(RewardConfig())

    def test_invalidated_prediction_does_not_affect_others(self):
        """La predicción de 5m del mismo feed sobrevive a la 15m invalidada."""
        quote = _nvda_quote()
        p15 = _prediction(quote, prediction_id="nvda-15m", horizon_seconds=900)
        p5 = _prediction(quote, prediction_id="nvda-5m", horizon_seconds=300)
        p15.invalidate()
        o5 = Outcome.from_prices(
            symbol="NVDA",
            ref_timestamp=TS,
            ref_price=100.00,
            horizon_seconds=300,
            final_price=100.60,
        )
        rewarded = p5.activate().to_waiting_outcome(o5).evaluate(o5).issue_reward(
            RewardConfig()
        )
        assert rewarded.status is PredictionStatus.REWARDED
        assert rewarded.reward.total > 0
