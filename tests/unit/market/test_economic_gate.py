"""FASE 6 — Gate económico duro: qué información tiene valor DESPUÉS
de costos. accuracy 58% perdiendo plata vs 51.5% ganando: manda el neto.

UNCALIBRATED → GROSS (bruto<=0) → COST (neto<=0) → probabilidad.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
from iot_machine_learning.domain.entities.market import Candle, DataStatus
from iot_machine_learning.domain.entities.market.calibration import (
    AdaptiveCalibrator,
    CalibrationMethod,
    ContextKey,
    EvidenceGate,
    GateReason,
    TradeAction,
    export_calibrator_state,
    try_refit,
)
from iot_machine_learning.domain.entities.market.calibration.pipeline import (
    UNCALIBRATED,
    CalibrationEvidence,
)
from iot_machine_learning.domain.entities.market.costs import (
    COST_PROFILES,
    CostModel,
)
from iot_machine_learning.domain.entities.market.costs_net import (
    dynamic_cost_for,
    evaluate_net,
    with_observed_spread,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.runners.paper_runner import (
    PaperBotConfig,
    PaperBotRunner,
)
from test_paper_runner import FakeEvidenceRepo, FakeFeed, FakePredictionRepo

BTC = COST_PROFILES["BTC-USD"]  # 24 bps
SPY = COST_PROFILES["SPY"]  # 12 bps


def _evidence(prob: float, fallback: str = "GLOBAL") -> CalibrationEvidence:
    return CalibrationEvidence(
        prediction_id="BTC-USD-momentum-3600-60",
        symbol="BTC-USD",
        horizon_seconds=60,
        regime="ALL",
        prob_raw=prob,
        prob_calibrated=prob,
        fallback_level=fallback,
        calibrator_version="v1",
        observation_timestamp=3600.0,
    )


class TestDynamicCosts:
    def test_spread_observado_reemplaza_ida_y_vuelta(self):
        assert with_observed_spread(BTC, 1.0).total_bps == 24
        assert with_observed_spread(BTC, 5.0).total_bps == 32  # 10+2+20
        assert with_observed_spread(SPY, 2.0).total_bps == 12  # 4+5+3

    def test_spread_invalido_falla(self):
        with pytest.raises(ValueError, match="inválido"):
            with_observed_spread(BTC, -1.0)
        with pytest.raises(TypeError, match="CostModel"):
            with_observed_spread("24bps", 1.0)

    def test_dynamic_cost_for_por_simbolo(self):
        assert dynamic_cost_for("BTC-USD", 1.0) == BTC
        assert dynamic_cost_for("SPY", 2.0) == SPY
        with pytest.raises(ValueError, match="sin perfil"):
            dynamic_cost_for("OVNI", 1.0)

    def test_evaluate_net_clasifica_la_escalera(self):
        assert evaluate_net(0.001, BTC).edge == "cost_negative"  # 10bps < 24
        assert evaluate_net(-0.001, BTC).edge == "gross_negative"
        assert evaluate_net(0.01, BTC).edge == "cost_positive"
        assert evaluate_net(0.01, BTC, sharpe=0.1).edge == "risk_negative"
        assert evaluate_net(0.01, BTC, sharpe=1.2).edge == "risk_adjusted_positive"
        evaluation = evaluate_net(0.01, BTC)
        assert evaluation.net_return == pytest.approx(0.01 - 0.0024)
        assert evaluation.cost_fraction == pytest.approx(0.0024)

    def test_evaluate_net_valida_entradas(self):
        with pytest.raises(TypeError, match="CostModel"):
            evaluate_net(0.01, "24bps")
        with pytest.raises(ValueError, match="no finito"):
            evaluate_net(float("nan"), BTC)


class TestEconomicGate:
    def test_calibrado_fuerte_pero_sin_neto_no_opera(self):
        # 0.80 calibrado con +10bps bruto en BTC (24bps): NO_TRADE honesto.
        decision = EvidenceGate().decide_with_costs(
            _evidence(0.80), expected_gross=0.001, cost_model=BTC)
        assert decision.action is TradeAction.NO_TRADE
        assert decision.reason is GateReason.COST_BLOCKED

    def test_sin_edge_ni_en_bruto(self):
        decision = EvidenceGate().decide_with_costs(
            _evidence(0.80), expected_gross=-0.002, cost_model=BTC)
        assert decision.action is TradeAction.NO_TRADE
        assert decision.reason is GateReason.GROSS_BLOCKED

    def test_neto_cero_es_bloqueo(self):
        # Frontera exacta: neto == 0 no paga el intento.
        decision = EvidenceGate().decide_with_costs(
            _evidence(0.80), expected_gross=BTC.total(), cost_model=BTC)
        assert decision.reason is GateReason.COST_BLOCKED
        gross_zero = EvidenceGate().decide_with_costs(
            _evidence(0.80), expected_gross=0.0, cost_model=BTC)
        assert gross_zero.reason is GateReason.GROSS_BLOCKED

    def test_con_neto_la_probabilidad_manda(self):
        gate = EvidenceGate()
        assert gate.decide_with_costs(
            _evidence(0.80), expected_gross=0.01,
            cost_model=BTC).reason is GateReason.LONG_SIGNAL
        assert gate.decide_with_costs(
            _evidence(0.20), expected_gross=0.01,
            cost_model=BTC).reason is GateReason.SHORT_SIGNAL
        assert gate.decide_with_costs(
            _evidence(0.52), expected_gross=0.01,
            cost_model=BTC).reason is GateReason.NEUTRAL_ZONE

    def test_uncalibrated_veta_antes_que_costos(self):
        decision = EvidenceGate().decide_with_costs(
            _evidence(0.99, fallback=UNCALIBRATED),
            expected_gross=0.50, cost_model=BTC)
        assert decision.reason is GateReason.UNCALIBRATED

    def test_spy_vs_btc_mismo_bruto_distinto_veredicto(self):
        # +20bps brutos: en SPY (12bps) opera, en BTC (24bps) no.
        gate = EvidenceGate()
        long_spy = gate.decide_with_costs(
            _evidence(0.80), expected_gross=0.002, cost_model=SPY)
        blocked_btc = gate.decide_with_costs(
            _evidence(0.80), expected_gross=0.002, cost_model=BTC)
        assert long_spy.action is TradeAction.LONG
        assert blocked_btc.reason is GateReason.COST_BLOCKED


class TestRunnerEconomicGate:
    def _artifact(self, tmp_path: Path) -> Path:
        rng = random.Random(7)
        ctx = ContextKey("momentum", 900, "ALL")
        pairs = [(ctx, 0.90, rng.random() < 0.50) for _ in range(600)]
        calibrator = AdaptiveCalibrator(method=CalibrationMethod.PLATT)
        calibrator.set_version("v4")
        assert try_refit(calibrator, pairs) is True
        artifact = tmp_path / "calibrator.json"
        artifact.write_text(json.dumps(export_calibrator_state(calibrator)))
        return artifact

    def _seed(self, start: float, n: int) -> list[Candle]:
        rng = random.Random(5)
        out, price = [], 100.0
        for i in range(n):
            price *= 1.0 + rng.uniform(-0.002, 0.004)
            out.append(Candle(
                symbol="BTC-USD", timestamp=start + i * 60,
                data_status=DataStatus.REALTIME, source_provider="fake",
                interval_seconds=60, open=price, high=price * 1.001,
                low=price * 0.999, close=round(price, 2), volume=1.0))
        return out

    def _run(self, seed, artifact, **config_overrides):
        pred_repo, ev_repo = FakePredictionRepo(), FakeEvidenceRepo()
        runner = PaperBotRunner(
            config=PaperBotConfig(
                window_candles=40, horizons_seconds=(60,),
                **config_overrides),
            feed=FakeFeed([seed]),
            prediction_repo_factory=lambda: pred_repo,
            evidence_repo_factory=lambda: ev_repo,
            on_status=lambda line: None,
            calibrator_state_path=artifact,
        )
        runner.run_cycle()
        return ev_repo.records[0]

    def test_simbolo_sin_perfil_falla_cerrado(self):
        with pytest.raises(ValueError, match="sin perfil de costos"):
            PaperBotRunner(
                config=PaperBotConfig(symbol="OVNI-USD"),
                feed=FakeFeed([[]]),
                prediction_repo_factory=FakePredictionRepo,
                evidence_repo_factory=FakeEvidenceRepo,
                on_status=lambda line: None,
            )

    def test_runner_bloquea_por_costos_con_artefacto(self, tmp_path: Path):
        seed = self._seed(3600.0, 40)
        record = self._run(seed, self._artifact(tmp_path))
        # Bruto +12.88bps < 24bps BTC: el gate económico veta.
        assert record.decision.action is TradeAction.NO_TRADE
        assert record.decision.reason is GateReason.COST_BLOCKED

    def test_flag_off_restaura_probabilidad_pura(self, tmp_path: Path):
        seed = self._seed(3600.0, 40)
        record = self._run(
            seed, self._artifact(tmp_path), require_positive_net=False)
        assert record.decision.reason is GateReason.NEUTRAL_ZONE
