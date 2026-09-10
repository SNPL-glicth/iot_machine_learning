"""FASE 4 — Anomalía / regime change: evitar operar también es operar.

Normal → volumen anormal + spread + imbalance + explosión de vol
→ DISTRIBUTION_SHIFT → NO_TRADE (gate) + dimensión de evidencia.
"""

from __future__ import annotations

import math

import pytest
from iot_machine_learning.domain.entities.market.anomaly import (
    SHIFT_QUORUM,
    SHIFT_WEIGHTS,
    binary_score,
    detect_shift,
    flow_extreme,
    robust_z,
    spread_shock,
    volatility_explosion,
    volume_spike,
)
from iot_machine_learning.domain.entities.market.calibration import (
    EvidenceGate,
    GateReason,
    TradeAction,
)
from iot_machine_learning.domain.entities.market.calibration.pipeline import (
    CalibrationEvidence,
)
from iot_machine_learning.domain.entities.market.evidence import EvidenceEngine
from iot_machine_learning.domain.entities.market.regime import cusum_changepoint


VOL_BASE = [100 + 5 * math.sin(i) for i in range(19)]
SPR_BASE = [10 + 0.5 * math.sin(i) for i in range(19)]
RET_CALM = [0.001 * math.sin(i) for i in range(45)]
RET_EXPL = [0.001 * math.sin(i) for i in range(40)] + [
    0.005 * math.sin(i) for i in range(5)
]


def _scores(*, shift: bool):
    last_vol = 500.0 if shift else 102.0
    last_spr = 40.0 if shift else 10.2
    rets = RET_EXPL if shift else RET_CALM
    imbs = (0.1, 0.2, 0.95) if shift else (0.1, 0.2, 0.3)
    cp = cusum_changepoint(
        tuple([0.001 * math.sin(i) for i in range(30)]
              + [0.008 + 0.001 * math.sin(i) for i in range(30)])
    ) is not None
    return (
        volume_spike(VOL_BASE + [last_vol]),
        spread_shock(SPR_BASE + [last_spr]),
        volatility_explosion(tuple(rets)),
        flow_extreme(imbs),
        binary_score("changepoint", cp if shift else False),
        binary_score("staleness", False),
    )


def _evidence(prob: float = 0.80) -> CalibrationEvidence:
    return CalibrationEvidence(
        prediction_id="BTC-USD-micro-l1-3600-60",
        symbol="BTC-USD",
        horizon_seconds=60,
        regime="TRENDING",
        prob_raw=prob,
        prob_calibrated=prob,
        fallback_level="GLOBAL",
        calibrator_version="v1",
        observation_timestamp=3600.0,
    )


class TestScores:
    def test_volumen_spike_vs_calma(self):
        assert volume_spike(VOL_BASE + [102.0]).breached is False
        spike = volume_spike(VOL_BASE + [500.0])
        assert spike.breached is True

    def test_spread_shock_vs_calma(self):
        assert spread_shock(SPR_BASE + [10.2]).breached is False
        assert spread_shock(SPR_BASE + [40.0]).breached is True

    def test_vol_explosion_ratio(self):
        calm = volatility_explosion(tuple(RET_CALM))
        assert calm.breached is False
        expl = volatility_explosion(tuple(RET_EXPL))
        assert expl.breached is True
        assert "4.34x" in expl.detail

    def test_flow_extreme(self):
        assert flow_extreme((0.1, 0.2, 0.3)).breached is False
        assert flow_extreme((0.1, 0.2, 0.95)).breached is True

    def test_robusto_a_outlier_en_baseline(self):
        # Un spike viejo en la baseline no ciega al detector (mediana/MAD
        # lo ignoran; media/desvío clásicos quedarían inflados).
        noisy = [100 + 5 * math.sin(i) for i in range(19)]
        series = noisy[:9] + [900.0] + noisy[9:] + [500.0]
        assert volume_spike(series).breached is True

    def test_baseline_constante_caso_borde(self):
        assert volume_spike([100.0] * 19 + [500.0]).breached is True
        assert volume_spike([100.0] * 20).breached is False

    def test_series_insuficientes_fallan(self):
        with pytest.raises(ValueError, match="insuficiente"):
            robust_z([1.0, 2.0])
        with pytest.raises(ValueError, match="insuficientes"):
            flow_extreme((0.1, 0.2))

    def test_binaria_desconocida_falla(self):
        with pytest.raises(ValueError, match="desconocida"):
            binary_score("ovnis", True)


class TestVote:
    def test_pesos_suman_uno(self):
        assert sum(SHIFT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_mercado_normal_no_shift(self):
        verdict = detect_shift(_scores(shift=False))
        assert verdict.is_shift is False
        assert verdict.n_breached == 0

    def test_mercado_anormal_shift_con_contribuyentes(self):
        verdict = detect_shift(_scores(shift=True))
        assert verdict.is_shift is True
        assert verdict.weight >= SHIFT_QUORUM
        assert "volume_spike" in verdict.contributors
        assert verdict.summary.startswith("SHIFT")

    def test_una_sola_senal_no_veta(self):
        scores = (
            volume_spike(VOL_BASE + [102.0]),
            spread_shock(SPR_BASE + [10.2]),
            volatility_explosion(tuple(RET_CALM)),
            flow_extreme((0.1, 0.2, 0.3)),
            binary_score("changepoint", True),  # solo 0.10 < 0.40
            binary_score("staleness", False),
        )
        assert detect_shift(scores).is_shift is False

    def test_duplicados_y_desconocidos_fallan(self):
        with pytest.raises(ValueError, match="duplicados"):
            detect_shift((volume_spike(VOL_BASE + [1.0]),
                          volume_spike(VOL_BASE + [1.0])))
        with pytest.raises(ValueError, match="sin scores"):
            detect_shift(())
        with pytest.raises(ValueError, match="quorum"):
            detect_shift(_scores(shift=False), quorum=1.5)


class TestGateVeto:
    def test_shift_veta_long_fuerte(self):
        gate = EvidenceGate()
        assert gate.decide(_evidence(0.80)).action is TradeAction.LONG
        vetoed = gate.decide_with_shift(_evidence(0.80), is_shift=True)
        assert vetoed.action is TradeAction.NO_TRADE
        assert vetoed.reason is GateReason.DISTRIBUTION_SHIFT

    def test_sin_shift_identico_a_decide(self):
        gate = EvidenceGate()
        for prob in (0.80, 0.20, 0.52):
            assert gate.decide_with_shift(_evidence(prob)) == gate.decide(_evidence(prob))

    def test_veredicto_real_veta(self):
        verdict = detect_shift(_scores(shift=True))
        decision = EvidenceGate().decide_with_shift(
            _evidence(0.80), is_shift=verdict.is_shift)
        assert decision.action is TradeAction.NO_TRADE


class TestEvidenceDimension:
    def _base_kwargs(self) -> dict:
        return dict(
            context="micro-l1·60s·TRENDING", n=200, history_days=10,
            accuracy=0.60, magnitude_errors=[0.005] * 200,
            expected_returns=[0.004] * 200, costs=[0.001] * 200,
            recency_accuracies=[0.60, 0.61],
            calibration_errors=[0.05] * 200,
        )

    def test_sin_shift_seis_dimensiones(self):
        verdict = EvidenceEngine().evaluate(**self._base_kwargs())
        assert len(verdict.dimensions) == 6

    def test_shift_tumba_evidencia(self):
        engine = EvidenceEngine()
        calm = engine.evaluate(**self._base_kwargs())
        assert calm.status.value == "evidence_supported"
        shifted = engine.evaluate(
            **self._base_kwargs(), shift_is_shift=True,
            shift_summary="SHIFT (peso 0.60/0.40: volume_spike,vol_explosion)",
        )
        assert len(shifted.dimensions) == 7
        assert shifted.status.value == "evidence_degraded"
        failed = [d.name for d in shifted.failed_dimensions]
        assert "distribution_shift" in failed
