"""Pruebas unitarias de FASE 8: PerformanceAnalyzer, WeightProposer y
AdaptationGuard. La premisa se prueba literal: ninguna propuesta toca el
modelo; el guard decide; y la razón es una cadena con números reales."""

from __future__ import annotations

import pytest
from iot_machine_learning.domain.entities.market.adaptation import (
    AdaptationGuard,
    ExpertScore,
    PerformanceAnalyzer,
    WeightProposal,
    WeightProposer,
    default_weights,
    wilson_lower_bound,
)

# ─── PerformanceAnalyzer ────────────────────────────────────────────────────


def _row(
    expert="momentum",
    regime="TRENDING",
    horizon=900,
    evaluated=50,
    hits=35,
    reward=40.0,
    calibration=0.05,
    days=3,
):
    return {
        "strategy": expert,
        "regime": regime,
        "horizon_seconds": horizon,
        "evaluated": evaluated,
        "hits": hits,
        "reward": reward,
        "calibration": calibration,
        "days": days,
    }


class TestPerformanceAnalyzer:
    def test_reward_adjusted_penalizes_bad_calibration(self):
        good = PerformanceAnalyzer().analyze((_row(calibration=0.05, reward=40.0),))[0]
        bad = PerformanceAnalyzer().analyze((_row(calibration=0.9, reward=40.0),))[0]
        assert good.reward_adjusted == pytest.approx(0.8 * (1 - 0.5 * 0.05))
        assert bad.reward_adjusted < good.reward_adjusted
        assert bad.calibration_error == 0.9

    def test_accuracy_does_not_enter_score(self):
        high = PerformanceAnalyzer().analyze((_row(hits=45),))[0]
        low = PerformanceAnalyzer().analyze((_row(hits=25),))[0]
        assert high.accuracy > low.accuracy
        assert high.reward_adjusted == low.reward_adjusted

    def test_rejects_invalid_rows(self):
        with pytest.raises(ValueError):
            PerformanceAnalyzer().analyze((_row(hits=51),))
        with pytest.raises(ValueError):
            PerformanceAnalyzer().analyze((_row(calibration=1.5),))
        with pytest.raises(ValueError):
            PerformanceAnalyzer().analyze((_row(evaluated=-1),))

    def test_penalty_validation(self):
        with pytest.raises(ValueError):
            PerformanceAnalyzer(calibration_penalty=1.5)
        with pytest.raises(ValueError):
            PerformanceAnalyzer(calibration_penalty=-0.1)

    def test_history_days_carried(self):
        score = PerformanceAnalyzer().analyze((_row(days=7),))[0]
        assert score.history_days == 7

    def test_accuracy_computed(self):
        score = PerformanceAnalyzer().analyze((_row(hits=35, evaluated=50),))[0]
        assert score.accuracy == 0.7


# ─── default_weights ────────────────────────────────────────────────────────


class TestDefaultWeights:
    def test_uniform(self):
        w = default_weights(("naive", "momentum", "ema"))
        assert w == {"naive": 1 / 3, "momentum": 1 / 3, "ema": 1 / 3}
        assert sum(w.values()) == pytest.approx(1.0)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            default_weights(())


# ─── WeightProposer ─────────────────────────────────────────────────────────


def _score(
    expert, regime="TRENDING", horizon=900, n=50, hits=35, reward=40.0, calibration=0.05, days=3
):
    return ExpertScore(
        expert=expert,
        regime=regime,
        horizon_seconds=horizon,
        n=n,
        accuracy=hits / n,
        mean_reward=reward / n,
        reward_total=reward,
        calibration_error=calibration,
        reward_adjusted=(reward / n) * (1 - 0.5 * min(calibration, 1.0)),
        history_days=days,
    )


class TestWeightProposer:
    def test_softmax_shift_toward_better_expert(self):
        proposer = WeightProposer(min_n=10)
        scores = (
            _score("momentum", reward=80.0, calibration=0.02),
            _score("naive", reward=10.0, calibration=0.05),
            _score("ema-crossover", reward=40.0, calibration=0.1),
        )
        current = {"*|TRENDING|900s": default_weights(("momentum", "naive", "ema-crossover"))}
        vector, proposals = proposer.propose_vector("TRENDING", 900, scores, current)
        assert sum(vector.values()) == pytest.approx(1.0)
        assert vector["momentum"] > current["*|TRENDING|900s"]["momentum"]
        assert vector["naive"] < current["*|TRENDING|900s"]["naive"]
        assert len(proposals) == 3

    def test_max_change_bounds(self):
        proposer = WeightProposer(min_n=10, max_change=0.05)
        scores = (
            _score("momentum", reward=80.0, calibration=0.02),
            _score("naive", reward=10.0, calibration=0.05),
            _score("ema-crossover", reward=40.0, calibration=0.1),
        )
        current = {"*|TRENDING|900s": default_weights(("momentum", "naive", "ema-crossover"))}
        vector, _ = proposer.propose_vector("TRENDING", 900, scores, current)
        for name, w in vector.items():
            assert abs(w - current["*|TRENDING|900s"][name]) <= 0.05 + 1e-9

    def test_min_n_gate(self):
        proposer = WeightProposer(min_n=100)
        current = {"*|TRENDING|900s": default_weights(("momentum", "naive"))}
        scores = (_score("momentum", n=50), _score("naive", n=50))
        vector, proposals = proposer.propose_vector("TRENDING", 900, scores, current)
        assert proposals == ()
        assert vector == current["*|TRENDING|900s"]  # sin muestra: nada cambia

    def test_no_expert_disappears(self):
        proposer = WeightProposer(min_n=10, min_weight=0.05)
        scores = (
            _score("momentum", reward=100.0, calibration=0.01),
            _score("naive", reward=5.0, calibration=0.6),
        )
        current = {"*|TRENDING|900s": {"momentum": 0.9, "naive": 0.1}}
        vector, _ = proposer.propose_vector("TRENDING", 900, scores, current)
        assert all(w >= 0.05 - 1e-9 for w in vector.values())
        assert sum(vector.values()) == pytest.approx(1.0)

    def test_proposal_fields_and_reason(self):
        proposer = WeightProposer(min_n=10)
        scores = (
            _score("momentum", reward=80.0, calibration=0.02),
            _score("naive", reward=10.0, calibration=0.05),
        )
        current = {"*|TRENDING|900s": default_weights(("momentum", "naive"))}
        _, proposals = proposer.propose_vector("TRENDING", 900, scores, current, parent_version="7")
        p = proposals[0]
        assert p.parent_version == "7"
        assert "increase" in p.reason and "900s" in p.reason
        assert p.sample_size == p.n if hasattr(p, "n") else p.sample_size == 50
        assert p.current_weight + p.weight_delta == pytest.approx(p.proposed_weight)

    def test_default_weight_for_unknown_experts(self):
        proposer = WeightProposer(min_n=10, default_weight=0.25)
        scores = (_score("momentum", reward=80.0), _score("naive", reward=10.0))
        current = {"*|TRENDING|900s": {"momentum": 0.6, "naive": 0.4}}
        vector, _ = proposer.propose_vector("TRENDING", 900, scores, current)
        assert set(vector) == {"momentum", "naive"}
        assert sum(vector.values()) == pytest.approx(1.0)

    def test_propose_flattens_contexts(self):
        proposer = WeightProposer(min_n=10)
        scores = (
            _score("momentum", regime="TRENDING", horizon=300, reward=80.0),
            _score("naive", regime="TRENDING", horizon=300, reward=10.0),
            _score("momentum", regime="RANGE", horizon=900, reward=80.0),
            _score("naive", regime="RANGE", horizon=900, reward=10.0),
        )
        current = {
            "*|TRENDING|300s": default_weights(("momentum", "naive")),
            "*|RANGE|900s": default_weights(("momentum", "naive")),
        }
        proposals = proposer.propose(scores, current)
        contexts = {p.context_label for p in proposals}
        assert any("TRENDING" in c and "300s" in c for c in contexts)
        assert any("RANGE" in c and "900s" in c for c in contexts)

    def test_validation(self):
        with pytest.raises(ValueError):
            WeightProposer(min_n=0)
        with pytest.raises(ValueError):
            WeightProposer(max_change=0.0)
        with pytest.raises(ValueError):
            WeightProposer(min_weight=0.0)
        with pytest.raises(ValueError):
            WeightProposer(temperature=0.0)


# ─── AdaptationGuard ────────────────────────────────────────────────────────


def _proposal(
    expert="momentum",
    regime="TRENDING",
    horizon=900,
    current=0.25,
    proposed=0.31,
    reward=0.81,
    calibration=0.02,
    n=183,
    accuracy=0.68,
    parent="7",
):
    return WeightProposal(
        expert=expert,
        regime=regime,
        horizon_seconds=horizon,
        current_weight=current,
        proposed_weight=proposed,
        observed_reward=reward,
        calibration=calibration,
        sample_size=n,
        accuracy=accuracy,
        reason="increase under TRENDING/900s (números reales)",
        created_at=1000.0,
        parent_version=parent,
    )


class TestAdaptationGuard:
    def test_accepts_healthy_proposal(self):
        guard = AdaptationGuard(min_n=10, min_history_days=2)
        result = guard.evaluate(
            _proposal(),
            history_days=3,
            context_weights_after={"momentum": 0.31, "naive": 0.69},
        )
        assert result.passed
        assert len(result.checks) == 9
        assert result.failed_checks == ()

    def test_rejects_small_sample(self):
        guard = AdaptationGuard(min_n=50)
        result = guard.evaluate(
            _proposal(n=20),
            history_days=3,
            context_weights_after={"momentum": 0.31, "naive": 0.69},
        )
        assert not result.passed
        assert any(c.name == "min_n" and not c.ok for c in result.checks)

    def test_rejects_insufficient_history(self):
        guard = AdaptationGuard(min_history_days=5)
        result = guard.evaluate(
            _proposal(),
            history_days=2,
            context_weights_after={"momentum": 0.31, "naive": 0.69},
        )
        assert not result.passed
        assert any(c.name == "history" and not c.ok for c in result.checks)

    def test_rejects_unclean_source(self):
        guard = AdaptationGuard()
        result = guard.evaluate(
            _proposal(),
            history_days=3,
            data_quality="has_stale",
            context_weights_after={"momentum": 0.31, "naive": 0.69},
        )
        assert not result.passed
        assert any(c.name == "clean_source" and not c.ok for c in result.checks)

    def test_rejects_invalid_reward(self):
        guard = AdaptationGuard()
        p = WeightProposal(
            expert="momentum",
            regime="TRENDING",
            horizon_seconds=900,
            current_weight=0.25,
            proposed_weight=0.31,
            observed_reward=float("nan"),
            calibration=0.02,
            sample_size=183,
            accuracy=0.68,
            reason="x",
            created_at=1.0,
            parent_version="7",
        )
        result = guard.evaluate(
            p, history_days=3, context_weights_after={"momentum": 0.31, "naive": 0.69}
        )
        assert not result.passed
        assert any(c.name == "reward_valid" and not c.ok for c in result.checks)

    def test_rejects_no_statistical_evidence(self):
        guard = AdaptationGuard()
        result = guard.evaluate(
            _proposal(accuracy=0.51, n=30),
            history_days=3,
            context_weights_after={"momentum": 0.31, "naive": 0.69},
        )
        assert not result.passed
        assert any(c.name == "statistical" and not c.ok for c in result.checks)

    def test_rejects_excessive_change(self):
        guard = AdaptationGuard(max_change=0.05)
        result = guard.evaluate(
            _proposal(current=0.25, proposed=0.40),
            history_days=3,
            context_weights_after={"momentum": 0.40, "naive": 0.60},
        )
        assert not result.passed
        assert any(c.name == "max_change" and not c.ok for c in result.checks)

    def test_rejects_broken_sum(self):
        guard = AdaptationGuard()
        result = guard.evaluate(
            _proposal(),
            history_days=3,
            context_weights_after={"momentum": 0.31, "naive": 0.60},
        )
        assert not result.passed
        assert any(c.name == "sum_weights" and not c.ok for c in result.checks)

    def test_rejects_disappearing_expert(self):
        guard = AdaptationGuard(min_weight=0.05)
        result = guard.evaluate(
            _proposal(),
            history_days=3,
            context_weights_after={"momentum": 1.0, "naive": 0.0},
        )
        assert not result.passed
        assert any(c.name == "min_weight" and not c.ok for c in result.checks)

    def test_rejects_missing_parent(self):
        guard = AdaptationGuard()
        p = _proposal(parent=None)
        result = guard.evaluate(
            p, history_days=3, context_weights_after={"momentum": 0.31, "naive": 0.69}
        )
        assert not result.passed
        assert any(c.name == "parent_preserved" and not c.ok for c in result.checks)

    def test_all_checks_reported(self):
        guard = AdaptationGuard(min_n=500)
        result = guard.evaluate(
            _proposal(n=30, accuracy=0.4, current=0.25, proposed=0.9),
            history_days=1,
            data_quality="dirty",
            context_weights_after={"momentum": 0.9},
        )
        assert not result.passed
        names = {c.name for c in result.checks}
        assert names == {
            "min_n",
            "history",
            "clean_source",
            "reward_valid",
            "statistical",
            "max_change",
            "sum_weights",
            "min_weight",
            "parent_preserved",
        }
        assert len(result.failed_checks) >= 5

    def test_validation(self):
        with pytest.raises(ValueError):
            AdaptationGuard(min_n=0)
        with pytest.raises(ValueError):
            AdaptationGuard(min_history_days=0)
        with pytest.raises(ValueError):
            AdaptationGuard(wilson_z=0.0)


# ─── wilson_lower_bound ─────────────────────────────────────────────────────


class TestWilson:
    def test_small_n_is_punished(self):
        assert wilson_lower_bound(6, 6) < 0.7  # 100% con n=6 no es "seguro"
        assert wilson_lower_bound(3, 6) < 0.4
        assert wilson_lower_bound(6, 6) > wilson_lower_bound(3, 6)

    def test_large_n_converges_to_p(self):
        assert wilson_lower_bound(700, 1000) > 0.67
        assert wilson_lower_bound(700, 1000) < 0.73

    def test_zero_n(self):
        assert wilson_lower_bound(0, 0) == 0.0

    def test_monotonic_in_hits(self):
        assert wilson_lower_bound(60, 100) > wilson_lower_bound(55, 100)
