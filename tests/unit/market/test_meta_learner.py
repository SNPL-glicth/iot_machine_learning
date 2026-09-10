"""FASE 5 — Meta-learner: cuándo confiar en cada estrategia.

REGIME A: momentum domina. REGIME B: mean-reversion domina.
Mismo learner, pesos distintos por contexto (Hedge sobre net
realizado, frío → uniforme, solo outcomes reales).
"""

from __future__ import annotations

import json

import pytest
from iot_machine_learning.domain.entities.market.adaptation import (
    META_STATE_VERSION,
    ExpertNetScore,
    ExpertOutcome,
    MetaLearner,
    SelectionConfig,
    SelectionMode,
    expert_net_scores,
    meta_adjusted_scores,
    select_weights,
)
from iot_machine_learning.domain.entities.market.adaptation.expert_scores import (
    ExpertScore,
)
from iot_machine_learning.domain.entities.market.costs import CostModel

EXPERTS = ("momentum", "mean-reversion", "ema-crossover", "micro-l1")


def _outcome(expert: str, regime: str, net: float,
             horizon: int = 300) -> ExpertOutcome:
    return ExpertOutcome(
        expert=expert, regime=regime, horizon_seconds=horizon, net_return=net,
    )


def _trained() -> MetaLearner:
    learner = MetaLearner(EXPERTS)
    outs = [_outcome("momentum", "TRENDING", 0.002) for _ in range(10)]
    outs += [_outcome("mean-reversion", "TRENDING", -0.001) for _ in range(10)]
    outs += [_outcome("mean-reversion", "RANGE", 0.002) for _ in range(10)]
    outs += [_outcome("momentum", "RANGE", -0.001) for _ in range(10)]
    learner.update(outs)
    return learner


def _net_scores() -> tuple[ExpertNetScore, ...]:
    scores = tuple(
        ExpertScore(
            expert=name, regime="TRENDING", horizon_seconds=300, n=50,
            accuracy=0.55, mean_reward=0.001, reward_total=0.05,
            calibration_error=0.1, reward_adjusted=0.0009, history_days=5,
            expected_return=0.002, realized_return=0.0015,
            execution_costs=0.0005, risk_std=0.001,
        )
        for name in EXPERTS
    )
    return expert_net_scores(scores, cost_model=CostModel(), min_n=10)


class TestMetaLearner:
    def test_frio_uniforme_explicito(self):
        weights = MetaLearner(EXPERTS).weights("TRENDING", 300)
        assert weights == {e: pytest.approx(0.25) for e in EXPERTS}

    def test_pesos_cambian_por_regimen(self):
        learner = _trained()
        trending = learner.weights("TRENDING", 300)
        ranging = learner.weights("RANGE", 300)
        assert trending["momentum"] > 0.5
        assert trending["momentum"] > trending["mean-reversion"]
        assert ranging["mean-reversion"] > 0.5
        assert ranging["mean-reversion"] > ranging["momentum"]
        assert sum(trending.values()) == pytest.approx(1.0)

    def test_regimen_no_visto_sigue_uniforme(self):
        learner = _trained()
        assert learner.weights("CRASH", 300) == {
            e: pytest.approx(0.25) for e in EXPERTS
        }

    def test_horizonte_separa_contextos(self):
        learner = MetaLearner(EXPERTS)
        learner.update([_outcome("momentum", "TRENDING", 0.01, horizon=60)])
        assert learner.weights("TRENDING", 60)["momentum"] > 0.25
        assert learner.weights("TRENDING", 300)["momentum"] == pytest.approx(0.25)

    def test_clip_evita_estado_absorbente(self):
        lucky = MetaLearner(EXPERTS)
        lucky.update([_outcome("momentum", "TRENDING", 50.0)])  # +5000%, recortado
        weights = lucky.weights("TRENDING", 300)
        assert weights["momentum"] > 0.25  # la evidencia fuerte pesa
        assert weights["momentum"] < 1.0  # pero nadie queda en cero
        assert min(weights.values()) > 0.0  # recuperable: soporte completo
        assert lucky.updates == 1

    def test_experto_desconocido_falla(self):
        with pytest.raises(ValueError, match="desconocido"):
            MetaLearner(EXPERTS).update([_outcome("ovni", "TRENDING", 0.01)])

    def test_outcome_invalido_falla(self):
        with pytest.raises(ValueError, match="vacío"):
            ExpertOutcome(expert="  ", regime="X", horizon_seconds=60,
                          net_return=0.01)
        with pytest.raises(ValueError, match="eta debe ser > 0"):
            MetaLearner(EXPERTS, eta=0.0)
        with pytest.raises(TypeError, match="ExpertOutcome"):
            MetaLearner(EXPERTS).update(["momentum"])

    def test_update_vacio_no_hace_nada(self):
        learner = MetaLearner(EXPERTS)
        assert learner.update([]) == 0
        assert learner.updates == 0


class TestStateRoundtrip:
    def test_json_versionable(self):
        learner = _trained()
        state = learner.to_state()
        assert state["version"] == META_STATE_VERSION
        restored = MetaLearner.from_state(json.loads(json.dumps(state)))
        assert restored.weights("TRENDING", 300) == learner.weights("TRENDING", 300)
        assert restored.weights("RANGE", 300) == learner.weights("RANGE", 300)
        assert restored.counts("TRENDING", 300) == learner.counts("TRENDING", 300)
        assert restored.updates == learner.updates

    def test_version_desconocida_falla(self):
        with pytest.raises(ValueError, match="versión"):
            MetaLearner.from_state({"version": "ovni-v9"})


class TestMetaAdjustedSelection:
    def test_uniforme_es_identidad(self):
        scores = _net_scores()
        uniform = {e: 0.25 for e in EXPERTS}
        adjusted = meta_adjusted_scores(scores, uniform)
        assert [s.score for s in adjusted] == pytest.approx(
            [s.score for s in scores])

    def test_sesgo_online_amplifica_ganador(self):
        scores = _net_scores()
        meta = _trained().weights("TRENDING", 300)
        adjusted = meta_adjusted_scores(scores, meta)
        by_name = {s.expert: s for s in adjusted}
        assert by_name["momentum"].score > by_name["mean-reversion"].score
        # Campos económicos intactos: el ajuste solo toca el score.
        assert by_name["momentum"].expected_net == [
            s for s in scores if s.expert == "momentum"][0].expected_net

    def test_cobertura_incompleta_falla(self):
        with pytest.raises(ValueError, match="cubrir"):
            meta_adjusted_scores(_net_scores(), {"momentum": 1.0})

    def test_flujo_completo_hasta_select_weights(self):
        learner = _trained()
        for regime, expected_winner in (("TRENDING", "momentum"),
                                        ("RANGE", "mean-reversion")):
            regime_scores = tuple(
                ExpertNetScore(
                    expert=s.expert, n=s.n, history_days=s.history_days,
                    expected_return=s.expected_return,
                    expected_cost=s.expected_cost,
                    risk_penalty=s.risk_penalty,
                    expected_net=s.expected_net,
                    calibration_quality=s.calibration_quality,
                    evidence_strength=s.evidence_strength, score=s.score,
                )
                for s in _net_scores()
            )
            adjusted = meta_adjusted_scores(
                regime_scores, learner.weights(regime, 300))
            result = select_weights(
                adjusted,
                config=SelectionConfig(mode=SelectionMode.SOFT,
                                       min_expected_net=-1.0),
            )
            assert result.weights[expected_winner] == max(
                result.weights.values())
