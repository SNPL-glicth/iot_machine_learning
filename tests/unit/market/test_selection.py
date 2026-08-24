"""FASE 9.4 — selección adaptativa de expertos (módulo puro).

La señal de ZENIN está en la SELECCIÓN (9.3), pero argmax a ciegas
amplifica ruido. Este módulo convierte ExpertScores en pesos con tres
modos (soft / selective / hard_max), guardrails y la puerta NO TRADE.
"""

import pytest
from iot_machine_learning.domain.entities.market.adaptation.expert_scores import ExpertScore
from iot_machine_learning.domain.entities.market.adaptation.selection import (
    DECISION_HOLD,
    DECISION_TRADE,
    ExpertNetScore,
    SelectionConfig,
    SelectionMode,
    expert_net_scores,
    select_weights,
)
from iot_machine_learning.domain.entities.market.costs import CostModel


def _score(
    expert: str,
    *,
    n: int = 40,
    days: int = 6,
    calibration: float = 0.1,
    expected: float = 0.004,
    risk_std: float = 0.012,
    accuracy: float = 0.6,
) -> ExpertScore:
    return ExpertScore(
        expert=expert,
        regime="trend_up",
        horizon_seconds=3600,
        n=n,
        accuracy=accuracy,
        mean_reward=0.003,
        reward_total=0.003 * n,
        calibration_error=calibration,
        reward_adjusted=0.003 * (1 - 0.5 * calibration),
        history_days=days,
        expected_return=expected,
        risk_std=risk_std,
    )


_STOCK = CostModel(spread_bps=4.0, slippage_bps=5.0, commission_bps=3.0)


class TestExpertNetScores:
    def test_expected_net_is_after_costs_and_risk(self):
        scores = expert_net_scores(
            [_score("A", expected=0.004, risk_std=0.012)],
            cost_model=_STOCK,
            risk_aversion=0.1,
            min_n=10,
        )
        net = scores[0]
        assert net.expected_cost == pytest.approx(0.0012)
        assert net.risk_penalty == pytest.approx(0.0012)
        assert net.expected_net == pytest.approx(0.004 - 0.0012 - 0.0012)
        assert net.calibration_quality == pytest.approx(0.9)
        assert net.evidence_strength == pytest.approx(1.0)

    def test_evidence_strength_capped_at_one(self):
        scores = expert_net_scores(
            [_score("A", n=100)], cost_model=_STOCK, min_n=10
        )
        assert scores[0].evidence_strength == pytest.approx(1.0)
        scores = expert_net_scores(
            [_score("A", n=5)], cost_model=_STOCK, min_n=10
        )
        assert scores[0].evidence_strength == pytest.approx(0.5)

    def test_score_multiplies_net_by_calibration_and_evidence(self):
        scores = expert_net_scores(
            [_score("A", n=5, calibration=0.5, expected=0.004, risk_std=0.012)],
            cost_model=_STOCK,
            risk_aversion=0.1,
            min_n=10,
        )
        net = scores[0]
        assert net.score == pytest.approx(
            net.expected_net * net.calibration_quality * net.evidence_strength
        )

    def test_accuracy_does_not_enter_score(self):
        hi = expert_net_scores(
            [_score("A", accuracy=0.9)], cost_model=_STOCK
        )[0]
        lo = expert_net_scores(
            [_score("A", accuracy=0.4)], cost_model=_STOCK
        )[0]
        assert hi.score == pytest.approx(lo.score)
        assert hi.score != 0.0

    def test_negative_expected_net_gives_negative_score(self):
        scores = expert_net_scores(
            [_score("A", expected=0.001)], cost_model=_STOCK, risk_aversion=0.1
        )
        assert scores[0].expected_net < 0.0
        assert scores[0].score < 0.0


class TestNoTradeGate:
    def test_hold_when_best_expected_net_not_positive(self):
        result = select_weights(
            expert_net_scores(
                [_score("A", expected=0.001)], cost_model=_STOCK, risk_aversion=0.1
            ),
            config=SelectionConfig(mode=SelectionMode.HARD_MAX),
        )
        assert result.decision == DECISION_HOLD
        assert result.weights == {}
        assert result.winner is None
        assert "edge neto" in result.reason

    def test_hold_when_no_scores_at_all(self):
        result = select_weights([], config=SelectionConfig(mode=SelectionMode.HARD_MAX))
        assert result.decision == DECISION_HOLD
        assert "sin expertos" in result.reason

    def test_trade_when_expected_net_positive(self):
        result = select_weights(
            expert_net_scores([_score("A")], cost_model=_STOCK),
            config=SelectionConfig(mode=SelectionMode.SOFT),
        )
        assert result.decision == DECISION_TRADE
        assert result.weights == {"A": 1.0}

    def test_custom_hold_threshold(self):
        result = select_weights(
            expert_net_scores([_score("A")], cost_model=_STOCK),
            config=SelectionConfig(mode=SelectionMode.SOFT, min_expected_net=0.003),
        )
        assert result.decision == DECISION_HOLD


class TestSoftMode:
    def test_softmax_weights_sum_to_one(self):
        scores = expert_net_scores(
            [
                _score("A", expected=0.004),
                _score("B", expected=0.003),
                _score("C", expected=0.002),
                _score("D", expected=0.001),
            ],
            cost_model=_STOCK,
        )
        result = select_weights(scores, config=SelectionConfig(mode=SelectionMode.SOFT))
        assert result.mode == SelectionMode.SOFT
        assert sum(result.weights.values()) == pytest.approx(1.0)
        assert all(w > 0.0 for w in result.weights.values())

    def test_soft_prefers_better_expert(self):
        scores = expert_net_scores(
            [_score("A", expected=0.004), _score("B", expected=0.001)],
            cost_model=_STOCK,
        )
        result = select_weights(scores, config=SelectionConfig(mode=SelectionMode.SOFT))
        assert result.weights["A"] > result.weights["B"]
        assert result.winner == "A"

    def test_soft_low_temperature_concentrates(self):
        scores = expert_net_scores(
            [_score("A", expected=0.004), _score("B", expected=0.001)],
            cost_model=_STOCK,
        )
        wide = select_weights(
            scores, config=SelectionConfig(mode=SelectionMode.SOFT, temperature=5.0)
        )
        sharp = select_weights(
            scores, config=SelectionConfig(mode=SelectionMode.SOFT, temperature=0.001)
        )
        assert sharp.weights["A"] > wide.weights["A"]
        assert sharp.weights["A"] > 0.9


class TestSelectiveMode:
    def test_drops_experts_below_ratio(self):
        scores = expert_net_scores(
            [
                _score("A", expected=0.006),
                _score("B", expected=0.0043),
                _score("C", expected=0.002),
            ],
            cost_model=_STOCK,
        )
        result = select_weights(
            scores,
            config=SelectionConfig(mode=SelectionMode.SELECTIVE, min_ratio=0.5),
        )
        assert result.mode == SelectionMode.SELECTIVE
        assert set(result.weights) == {"A", "B"}
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_caps_experts_by_max_experts(self):
        scores = expert_net_scores(
            [_score(f"E{i}", expected=0.004 - 0.0001 * i) for i in range(4)],
            cost_model=_STOCK,
        )
        result = select_weights(
            scores,
            config=SelectionConfig(mode=SelectionMode.SELECTIVE, max_experts=2),
        )
        assert len(result.weights) == 2
        assert result.winner == "E0"

    def test_falls_back_to_soft_without_evidence(self):
        scores = expert_net_scores(
            [_score("A", n=3, days=1), _score("B", n=3, days=1)],
            cost_model=_STOCK,
        )
        result = select_weights(
            scores, config=SelectionConfig(mode=SelectionMode.SELECTIVE)
        )
        assert result.mode == SelectionMode.SOFT
        assert len(result.weights) == 2


class TestHardMaxMode:
    def test_winner_takes_all_with_guardrails(self):
        scores = expert_net_scores(
            [_score("A", expected=0.004), _score("B", expected=0.003)],
            cost_model=_STOCK,
        )
        result = select_weights(
            scores, config=SelectionConfig(mode=SelectionMode.HARD_MAX)
        )
        assert result.mode == SelectionMode.HARD_MAX
        assert result.weights == {"A": 1.0}
        assert "guardrails" in result.reason

    def test_falls_back_to_selective_when_margin_fails(self):
        scores = expert_net_scores(
            [_score("A", expected=0.004), _score("B", expected=0.0039)],
            cost_model=_STOCK,
        )
        result = select_weights(
            scores,
            config=SelectionConfig(mode=SelectionMode.HARD_MAX, min_margin=0.001),
        )
        assert result.mode == SelectionMode.SELECTIVE
        assert set(result.weights) == {"A", "B"}

    def test_falls_back_to_soft_without_evidence(self):
        scores = expert_net_scores(
            [_score("A", n=3, days=1), _score("B", n=3, days=1)],
            cost_model=_STOCK,
        )
        result = select_weights(
            scores, config=SelectionConfig(mode=SelectionMode.HARD_MAX)
        )
        assert result.mode == SelectionMode.SOFT

    def test_margin_passes_with_single_expert(self):
        scores = expert_net_scores(
            [_score("A", expected=0.004)], cost_model=_STOCK
        )
        result = select_weights(
            scores,
            config=SelectionConfig(mode=SelectionMode.HARD_MAX, min_margin=0.5),
        )
        assert result.mode == SelectionMode.HARD_MAX
        assert result.weights == {"A": 1.0}


class TestConfigValidation:
    def test_invalid_configs_rejected(self):
        with pytest.raises(ValueError):
            SelectionConfig(temperature=0.0)
        with pytest.raises(ValueError):
            SelectionConfig(min_ratio=1.5)
        with pytest.raises(ValueError):
            SelectionConfig(max_experts=0)
        with pytest.raises(ValueError):
            SelectionConfig(min_n=0)
        with pytest.raises(ValueError):
            SelectionConfig(risk_aversion=-0.1)


class TestSelectionResultAudit:
    def test_scores_are_ranked_in_result(self):
        scores = expert_net_scores(
            [_score("A", expected=0.004), _score("B", expected=0.001)],
            cost_model=_STOCK,
        )
        result = select_weights(scores, config=SelectionConfig(mode=SelectionMode.SOFT))
        assert [s.expert for s in result.scores] == ["A", "B"]

    def test_expert_net_score_fields(self):
        net = ExpertNetScore(
            expert="A", n=40, history_days=6,
            expected_return=0.004, expected_cost=0.0012,
            risk_penalty=0.0012, expected_net=0.0016,
            calibration_quality=0.9, evidence_strength=1.0,
            score=0.00144,
        )
        assert net.expert == "A"
        assert net.expected_net == pytest.approx(0.0016)
