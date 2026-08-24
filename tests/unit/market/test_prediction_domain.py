"""Tests unitarios — dominio de predicción (FASE 3).

Cubre el ciclo Prediction -> Outcome -> Evaluation -> Reward en memoria:
creación, validaciones, lifecycle, evaluación, reward multi-dimensión,
multi-horizonte independiente y la regla crítica "solo EVALUATED -> REWARDED
produce reward".
"""

from __future__ import annotations

import dataclasses
import math

import pytest
from iot_machine_learning.domain.entities.market import DataStatus, Quote
from iot_machine_learning.domain.entities.market.prediction import (
    Evaluation,
    InputContext,
    InvalidTransitionError,
    Outcome,
    Prediction,
    PredictionInterval,
    PredictionStatus,
    Regime,
    Reward,
    RewardConfig,
    compute_reward,
    evaluate_prediction,
)

TS = 1_600_000_000.0
MIDPOINT = 200.05


def _quote(**overrides) -> Quote:
    kwargs = dict(
        symbol="NVDA",
        timestamp=TS,
        data_status=DataStatus.REALTIME,
        source_provider="alpaca",
        bid=200.0,
        bid_size=5.0,
        ask=200.1,
        ask_size=7.0,
    )
    kwargs.update(overrides)
    return Quote(**kwargs)


def _prediction(**overrides) -> Prediction:
    kwargs = dict(
        prediction_id="p-1m",
        observation=_quote(),
        horizon_seconds=60,
        timestamp=TS,
        entry_price=MIDPOINT,
        expected_return=0.02,
        probability_up=0.7,
        confidence=0.8,
        interval=PredictionInterval(lower=-0.01, upper=0.05, confidence_level=0.90),
        regime=Regime.BULL,
        strategy="trend-v1",
    )
    kwargs.update(overrides)
    return Prediction(**kwargs)


def _outcome(prediction: Prediction, *, return_realized: float, **overrides) -> Outcome:
    kwargs = dict(
        symbol=prediction.observation.symbol,
        observation_timestamp=prediction.observation.timestamp,
        horizon_seconds=prediction.horizon_seconds,
        final_price=prediction.entry_price * (1.0 + return_realized),
        return_realized=return_realized,
    )
    kwargs.update(overrides)
    kwargs["measured_at"] = kwargs.get(
        "measured_at", kwargs["observation_timestamp"] + kwargs["horizon_seconds"]
    )
    return Outcome(**kwargs)


def _evaluated(prediction: Prediction, return_realized: float) -> Prediction:
    """Lleva una Prediction hasta EVALUATED con el desenlace dado."""
    outcome = _outcome(prediction, return_realized=return_realized)
    return (
        prediction.activate()
        .to_waiting_outcome(outcome)
        .evaluate(outcome)
    )


def _rewarded(prediction: Prediction, return_realized: float, config=None) -> Prediction:
    return _evaluated(prediction, return_realized).issue_reward(config or RewardConfig())


# ─── Creación y validación ────────────────────────────────────────────────


class TestPredictionCreation:
    def test_valid_creation(self):
        p = _prediction()
        assert p.prediction_id == "p-1m"
        assert p.status is PredictionStatus.PENDING
        assert p.observation.symbol == "NVDA"
        assert p.interval.contains(0.02)
        assert p.created_at <= p.updated_at

    @pytest.mark.parametrize("horizon", [0, -60, -1])
    def test_invalid_horizon_rejected(self, horizon):
        with pytest.raises(ValueError):
            _prediction(horizon_seconds=horizon)

    def test_non_int_horizon_rejected(self):
        with pytest.raises(TypeError):
            _prediction(horizon_seconds=60.5)

    @pytest.mark.parametrize("probability_up", [-0.1, 1.5, float("nan"), float("inf")])
    def test_probability_out_of_range_rejected(self, probability_up):
        with pytest.raises(ValueError):
            _prediction(probability_up=probability_up)

    @pytest.mark.parametrize("confidence", [-0.5, 2.0, float("nan")])
    def test_confidence_out_of_range_rejected(self, confidence):
        with pytest.raises(ValueError):
            _prediction(confidence=confidence)

    @pytest.mark.parametrize("lower,upper", [(0.05, -0.01), (0.02, 0.02)])
    def test_incoherent_interval_rejected(self, lower, upper):
        with pytest.raises(ValueError):
            _prediction(interval=PredictionInterval(lower=lower, upper=upper))

    def test_interval_not_containing_expected_return_rejected(self):
        with pytest.raises(ValueError):
            _prediction(
                expected_return=0.10,
                interval=PredictionInterval(lower=-0.01, upper=0.05),
            )

    def test_invalid_probability_interval_level(self):
        with pytest.raises(ValueError):
            PredictionInterval(lower=-0.01, upper=0.05, confidence_level=1.5)

    def test_entry_price_invalid_rejected(self):
        with pytest.raises(ValueError):
            _prediction(entry_price=0.0)

    def test_empty_prediction_id_rejected(self):
        with pytest.raises(ValueError):
            _prediction(prediction_id="   ")

    def test_invalid_status_type_rejected(self):
        with pytest.raises(TypeError):
            _prediction(status="active")  # type: ignore[arg-type]

    def test_impossible_states_rejected(self):
        with pytest.raises(ValueError):
            _prediction(status=PredictionStatus.WAITING_OUTCOME)  # sin outcome
        with pytest.raises(ValueError):
            _prediction(
                status=PredictionStatus.REWARDED,
                outcome=_outcome(_prediction(), return_realized=0.0),
                evaluation=Evaluation(
                    direction_correct=True, magnitude_error=0.0,
                    within_interval=True, calibration_error=0.3,
                ),
            )  # reward faltante

    def test_invalid_context_rejected(self):
        with pytest.raises(ValueError):
            _prediction(input_context=InputContext(feature_count=-3))
        with pytest.raises(TypeError):
            _prediction(input_context=InputContext(data_status="stale"))  # type: ignore[arg-type]

    def test_immutability(self):
        p = _prediction()
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.expected_return = 0.99  # type: ignore[misc]
        with pytest.raises(dataclasses.FrozenInstanceError):
            p.confidence = 0.1  # type: ignore[misc]


# ─── Timestamps consistentes ───────────────────────────────────────────────


class TestTimestampsConsistent:
    def test_prediction_cannot_predate_observation(self):
        with pytest.raises(ValueError):
            _prediction(timestamp=TS - 1.0)

    def test_outcome_cannot_be_measured_before_horizon_end(self):
        p = _prediction()
        with pytest.raises(ValueError):
            _outcome(p, return_realized=0.01, measured_at=TS + 59.0)

    def test_outcome_factory_measures_at_horizon_end_by_default(self):
        o = Outcome.from_prices(
            symbol="NVDA",
            ref_timestamp=TS,
            ref_price=MIDPOINT,
            horizon_seconds=300,
            final_price=MIDPOINT * 1.01,
        )
        assert o.measured_at == TS + 300
        assert o.return_realized == pytest.approx(0.01)

    def test_updated_at_progresses_through_cycle(self):
        p = _rewarded(_prediction(), return_realized=0.02)
        assert p.created_at <= p.updated_at


# ─── Lifecycle ─────────────────────────────────────────────────────────────


class TestLifecycle:
    def test_full_cycle(self):
        p = _prediction()
        a = p.activate()
        assert a.status is PredictionStatus.ACTIVE
        o = _outcome(a, return_realized=0.02)
        w = a.to_waiting_outcome(o)
        assert w.status is PredictionStatus.WAITING_OUTCOME
        assert w.outcome is o
        e = w.evaluate(o)
        assert e.status is PredictionStatus.EVALUATED
        assert e.evaluation is not None
        r = e.issue_reward(RewardConfig())
        assert r.status is PredictionStatus.REWARDED
        assert r.reward is not None
        ar = r.archive()
        assert ar.status is PredictionStatus.ARCHIVED

    @pytest.mark.parametrize(
        "target_status",
        [
            PredictionStatus.EVALUATED,
            PredictionStatus.REWARDED,
            PredictionStatus.WAITING_OUTCOME,
        ],
    )
    def test_pending_cannot_jump_states(self, target_status):
        p = _prediction()
        o = _outcome(p, return_realized=0.0)
        with pytest.raises(InvalidTransitionError):
            p.evaluate(o)
        with pytest.raises(InvalidTransitionError):
            p.issue_reward(RewardConfig())

    def test_active_cannot_reward(self):
        p = _prediction().activate()
        with pytest.raises(InvalidTransitionError):
            p.issue_reward(RewardConfig())

    def test_waiting_outcome_cannot_reward(self):
        p = _prediction()
        o = _outcome(p, return_realized=0.02)
        w = p.activate().to_waiting_outcome(o)
        with pytest.raises(InvalidTransitionError):
            w.issue_reward(RewardConfig())

    def test_invalidated_is_terminal(self):
        p = _prediction().invalidate()
        assert p.status is PredictionStatus.INVALIDATED
        with pytest.raises(InvalidTransitionError):
            p.activate()
        with pytest.raises(InvalidTransitionError):
            p.evaluate(_outcome(p, return_realized=0.0))
        with pytest.raises(InvalidTransitionError):
            p.issue_reward(RewardConfig())

    def test_archived_is_terminal(self):
        p = _rewarded(_prediction(), return_realized=0.02).archive()
        assert p.status is PredictionStatus.ARCHIVED
        with pytest.raises(InvalidTransitionError):
            p.activate()

    def test_invalidate_from_waiting_outcome(self):
        p = _prediction()
        w = p.activate().to_waiting_outcome(_outcome(p, return_realized=0.02))
        assert w.invalidate().status is PredictionStatus.INVALIDATED

    def test_invalidated_cannot_rewind(self):
        p = _prediction().invalidate()
        with pytest.raises(InvalidTransitionError):
            p.archive()

def test_can_produce_reward_flag():
    # PENDING: sin evaluación no hay reward posible (contrato del ciclo de
    # vida: solo EVALUATED -> REWARDED lo materializa).
    assert not _prediction().can_produce_reward
    assert not _prediction().invalidate().can_produce_reward
    w = _prediction().activate().to_waiting_outcome(
        _outcome(_prediction(), return_realized=0.02)
    )
    assert not w.can_produce_reward
    e = _evaluated(_prediction(), return_realized=0.02)
    assert e.can_produce_reward
    assert not e.archive().can_produce_reward


# ─── Outcome ───────────────────────────────────────────────────────────────


class TestOutcome:
    def test_outcome_horizon_mismatch_rejected(self):
        p = _prediction(horizon_seconds=60)
        wrong = _outcome(p, return_realized=0.01, horizon_seconds=300)
        with pytest.raises(ValueError, match="horizon"):
            p.activate().to_waiting_outcome(wrong)

    def test_outcome_symbol_mismatch_rejected(self):
        p = _prediction()
        wrong = _outcome(p, return_realized=0.01, symbol="TSLA")
        with pytest.raises(ValueError, match="símbolo"):
            p.activate().to_waiting_outcome(wrong)

    def test_outcome_invalid_price_rejected(self):
        with pytest.raises(ValueError):
            Outcome.from_prices(
                symbol="NVDA",
                ref_timestamp=TS,
                ref_price=0.0,
                horizon_seconds=60,
                final_price=10.0,
            )


# ─── Evaluación ────────────────────────────────────────────────────────────


class TestEvaluation:
    def test_direction_correct_when_up(self):
        e = evaluate_prediction(
            _prediction(expected_return=0.02, probability_up=0.7),
            _outcome(_prediction(expected_return=0.02), return_realized=0.025),
        )
        assert e.direction_correct is True

    def test_direction_incorrect_when_down(self):
        e = evaluate_prediction(
            _prediction(expected_return=0.02),
            _outcome(_prediction(expected_return=0.02), return_realized=-0.015),
        )
        assert e.direction_correct is False

    def test_direction_for_down_prediction(self):
        e = evaluate_prediction(
            _prediction(expected_return=-0.02, interval=None),
            _outcome(_prediction(expected_return=-0.02, interval=None), return_realized=-0.03),
        )
        assert e.direction_correct is True

    def test_magnitude_error(self):
        e = evaluate_prediction(
            _prediction(expected_return=0.02),
            _outcome(_prediction(expected_return=0.02), return_realized=0.025),
        )
        assert e.magnitude_error == pytest.approx(0.005)

    def test_within_interval(self):
        e = evaluate_prediction(
            _prediction(),
            _outcome(_prediction(), return_realized=0.03),
        )
        assert e.within_interval is True
        e2 = evaluate_prediction(
            _prediction(),
            _outcome(_prediction(), return_realized=0.08),
        )
        assert e2.within_interval is False

    def test_within_interval_false_without_interval(self):
        e = evaluate_prediction(
            _prediction(interval=None),
            _outcome(_prediction(interval=None), return_realized=0.02),
        )
        assert e.within_interval is False

    def test_calibration_error(self):
        e = evaluate_prediction(
            _prediction(probability_up=0.7),
            _outcome(_prediction(probability_up=0.7), return_realized=0.02),
        )
        assert e.calibration_error == pytest.approx(0.3)
        e2 = evaluate_prediction(
            _prediction(probability_up=0.7),
            _outcome(_prediction(probability_up=0.7), return_realized=-0.02),
        )
        assert e2.calibration_error == pytest.approx(0.7)
        e3 = evaluate_prediction(
            _prediction(probability_up=1.0),
            _outcome(_prediction(probability_up=1.0), return_realized=0.02),
        )
        assert e3.calibration_error == pytest.approx(0.0)

    def test_flat_return_calibrates_as_down(self):
        e = evaluate_prediction(
            _prediction(probability_up=0.5),
            _outcome(_prediction(probability_up=0.5), return_realized=0.0),
        )
        assert e.calibration_error == pytest.approx(0.5)

    def test_evaluation_surfaces_through_prediction(self):
        r = _evaluated(_prediction(), return_realized=0.025)
        assert r.evaluation.direction_correct is True
        assert r.evaluation.magnitude_error == pytest.approx(0.005)


# ─── Reward ────────────────────────────────────────────────────────────────


class TestReward:
    def test_reward_positive_favorable(self):
        p = _rewarded(_prediction(expected_return=0.02, probability_up=1.0),
                      return_realized=0.02)
        assert p.reward.total > 0.0

    def test_reward_negative_unfavorable(self):
        p = _rewarded(_prediction(expected_return=0.02, probability_up=0.7),
                      return_realized=-0.05)
        assert p.reward.total < 0.0

    def test_reward_components(self):
        r = _rewarded(_prediction(expected_return=0.02, probability_up=1.0),
                      return_realized=0.02).reward
        assert r.direction_component == pytest.approx(1.0)
        assert r.magnitude_component == pytest.approx(0.5)
        assert r.calibration_component == pytest.approx(0.25)
        assert r.execution_costs == pytest.approx(0.0)
        assert r.total == pytest.approx(1.75)

    def test_magnitude_component_degrades_with_error(self):
        cfg = RewardConfig()
        r_good = compute_reward(
            _prediction(expected_return=0.02),
            _outcome(_prediction(expected_return=0.02), return_realized=0.02),
            evaluate_prediction(
                _prediction(expected_return=0.02),
                _outcome(_prediction(expected_return=0.02), return_realized=0.02),
            ),
            cfg,
        )
        r_bad = compute_reward(
            _prediction(expected_return=0.02),
            _outcome(_prediction(expected_return=0.02), return_realized=0.04),
            evaluate_prediction(
                _prediction(expected_return=0.02),
                _outcome(_prediction(expected_return=0.02), return_realized=0.04),
            ),
            cfg,
        )
        assert r_bad.magnitude_component < r_good.magnitude_component
        assert r_bad.total < r_good.total

    def test_costs_slippage_risk_penalty_reduce_reward(self):
        base = _rewarded(_prediction(expected_return=0.02, probability_up=1.0),
                         return_realized=0.02)
        costly = _rewarded(
            _prediction(expected_return=0.02, probability_up=1.0),
            return_realized=0.02,
            config=RewardConfig(cost_rate=0.10, slippage_rate=0.05, risk_penalty=0.05),
        )
        assert costly.reward.execution_costs == pytest.approx(0.20)
        assert costly.reward.total == pytest.approx(base.reward.total - 0.20)

    def test_reward_zero_expected_return_no_magnitude_credit(self):
        r = compute_reward(
            _prediction(expected_return=0.0),
            _outcome(_prediction(expected_return=0.0), return_realized=0.0),
            evaluate_prediction(
                _prediction(expected_return=0.0),
                _outcome(_prediction(expected_return=0.0), return_realized=0.0),
            ),
            RewardConfig(),
        )
        assert r.magnitude_component == pytest.approx(0.0)
        assert r.direction_component == pytest.approx(1.0)

    def test_invalid_reward_config_rejected(self):
        with pytest.raises(ValueError):
            RewardConfig(direction_weight=-1.0)
        with pytest.raises(ValueError):
            RewardConfig(cost_rate=float("nan"))

    def test_reward_is_immutable(self):
        r = _rewarded(_prediction(), return_realized=0.02).reward
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.total = 99.0  # type: ignore[misc]


# ─── Múltiples horizontes independientes ───────────────────────────────────


class TestMultipleHorizons:
    def _three_horizons(self):
        return (
            _prediction(prediction_id="p-1m", horizon_seconds=60),
            _prediction(prediction_id="p-5m", horizon_seconds=300),
            _prediction(prediction_id="p-15m", horizon_seconds=900),
        )

    def test_three_cycles_independent(self):
        p1, p5, p15 = self._three_horizons()

        o1 = Outcome.from_prices(
            symbol="NVDA", ref_timestamp=TS, ref_price=MIDPOINT,
            horizon_seconds=60, final_price=MIDPOINT * 1.002,
        )
        p1_done = p1.activate().to_waiting_outcome(o1).evaluate(o1).issue_reward(
            RewardConfig()
        )
        assert p1_done.status is PredictionStatus.REWARDED
        assert p1_done.reward is not None

        assert p5.status is PredictionStatus.PENDING
        assert p5.outcome is None and p5.reward is None
        assert p15.status is PredictionStatus.PENDING

        p5_active = p5.activate()
        assert p5_active.status is PredictionStatus.ACTIVE
        assert p15.status is PredictionStatus.PENDING

    def test_1m_outcome_cannot_close_15m(self):
        p1, p5, p15 = self._three_horizons()
        o1 = Outcome.from_prices(
            symbol="NVDA", ref_timestamp=TS, ref_price=MIDPOINT,
            horizon_seconds=60, final_price=MIDPOINT * 1.002,
        )
        p15_active = p15.activate()
        with pytest.raises(ValueError, match="horizon"):
            p15_active.to_waiting_outcome(o1)
        assert p15_active.status is PredictionStatus.ACTIVE
        assert p15_active.outcome is None

    def test_15m_still_awaiting_after_1m_rewarded(self):
        p1, p5, p15 = self._three_horizons()
        o1 = Outcome.from_prices(
            symbol="NVDA", ref_timestamp=TS, ref_price=MIDPOINT,
            horizon_seconds=60, final_price=MIDPOINT * 1.002,
        )
        o15 = Outcome.from_prices(
            symbol="NVDA", ref_timestamp=TS, ref_price=MIDPOINT,
            horizon_seconds=900, final_price=MIDPOINT * 1.01,
        )
        p1_done = p1.activate().to_waiting_outcome(o1).evaluate(o1).issue_reward(
            RewardConfig()
        )
        p15_waiting = p15.activate().to_waiting_outcome(o15)
        assert p15_waiting.status is PredictionStatus.WAITING_OUTCOME
        assert p1_done.reward is not None
        assert p15_waiting.reward is None

    def test_identical_descriptions_do_not_share_state(self):
        a = _prediction(prediction_id="p-1m", horizon_seconds=60)
        b = _prediction(prediction_id="p-5m", horizon_seconds=300)
        assert a is not b
        assert a.status is b.status  # ambos PENDING, pero objetos distintos


# ─── Verificación final: matemática del reward con casos reales ────────────


class TestRewardMathReference:
    """Escenarios de referencia documentados en el contrato v1."""

    def test_bullish_prediction_hit(self):
        p = _prediction(expected_return=0.02, probability_up=0.7, confidence=0.8)
        r = _rewarded(p, return_realized=0.025).reward
        assert r.direction_component > 0
        assert r.total > 0

    def test_bullish_prediction_missed(self):
        p = _prediction(expected_return=0.02, probability_up=0.7)
        r = _rewarded(p, return_realized=-0.025).reward
        assert r.direction_component < 0
        assert r.total < 0

    def test_calibration_dominated_interval(self):
        cfg = RewardConfig(calibration_weight=2.0)
        r = _rewarded(
            _prediction(expected_return=0.02, probability_up=0.9),
            return_realized=-0.01,
            config=cfg,
        ).reward
        assert r.calibration_component == pytest.approx(2.0 * (1 - 0.9))

    def test_reward_shape_is_reward(self):
        r = _rewarded(_prediction(), return_realized=0.02).reward
        assert isinstance(r, Reward)
        assert math.isfinite(r.total)
