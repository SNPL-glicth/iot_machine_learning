"""FASE 1 — Predicción condicional distribucional.

Estado → distribución futura: expected return, volatility,
P(exceed +x%), P(drawdown), holding esperado, tail risk.
Métricas: pinball, CRPS, cobertura. Calibración por cuantil.
"""

from __future__ import annotations

import math

import pytest
from iot_machine_learning.domain.entities.market import DataStatus, Quote
from iot_machine_learning.domain.entities.market.calibration import (
    apply_quantile_shifts,
    empirical_coverage,
    fit_quantile_shift,
)
from iot_machine_learning.domain.entities.market.prediction import (
    DISTRIBUTIONAL_HORIZONS,
    Outcome,
    Prediction,
    ReturnDistribution,
    crps_gaussian,
    evaluate_prediction,
    pinball_loss,
)

TS = 1_600_000_000.0


def _dist(**overrides) -> ReturnDistribution:
    kwargs = dict(
        expected_return=0.0018,
        volatility=0.01,
        quantile_10=-0.011,
        quantile_50=0.0018,
        quantile_90=0.0146,
        expected_holding_seconds=300,
        cvar_05=-0.020,
    )
    kwargs.update(overrides)
    return ReturnDistribution(**kwargs)


def _prediction(dist: ReturnDistribution | None = None, **overrides) -> Prediction:
    quote = Quote(
        symbol="SPY", timestamp=TS, data_status=DataStatus.REALTIME,
        source_provider="alpaca", bid=500.0, bid_size=5.0,
        ask=500.1, ask_size=7.0,
    )
    expected = dist.expected_return if dist is not None else 0.0018
    kwargs = dict(
        prediction_id="spy-5m",
        observation=quote,
        horizon_seconds=300,
        timestamp=TS,
        entry_price=500.05,
        expected_return=expected,
        probability_up=0.57,
        confidence=0.8,
        distribution=dist,
        strategy="ml-v1",
    )
    kwargs.update(overrides)
    return Prediction(**kwargs)


def _outcome(prediction: Prediction, return_realized: float) -> Outcome:
    return Outcome(
        symbol=prediction.observation.symbol,
        observation_timestamp=prediction.observation.timestamp,
        horizon_seconds=prediction.horizon_seconds,
        measured_at=prediction.observation.timestamp + prediction.horizon_seconds,
        final_price=prediction.entry_price * (1.0 + return_realized),
        return_realized=return_realized,
    )


class TestReturnDistribution:
    def test_p_exceed_simetrica_es_half(self):
        dist = ReturnDistribution(
            expected_return=0.0, volatility=1.0,
            quantile_10=-1.2816, quantile_50=0.0, quantile_90=1.2816,
        )
        assert dist.p_exceed(0.0) == pytest.approx(0.5)
        assert dist.implied_probability_up == pytest.approx(0.5)
        assert dist.p_exceed(1.0) == pytest.approx(1.0 - 0.8413, abs=1e-3)

    def test_p_drawdown_es_cola_izquierda(self):
        dist = _dist()
        assert dist.p_drawdown(0.01) == pytest.approx(1.0 - dist.p_exceed(-0.01))
        with pytest.raises(ValueError, match="threshold debe ser > 0"):
            dist.p_drawdown(0.0)

    def test_volatilidad_no_positiva_falla(self):
        with pytest.raises(ValueError, match="volatility debe ser > 0"):
            _dist(volatility=0.0)

    def test_cuantiles_desordenados_fallan(self):
        with pytest.raises(ValueError, match="cuantiles desordenados"):
            _dist(quantile_50=0.99)

    def test_media_fuera_de_banda_falla(self):
        with pytest.raises(ValueError, match="fuera de la banda"):
            _dist(expected_return=0.50)

    def test_within_band(self):
        dist = _dist()
        assert dist.within_band(0.0)
        assert not dist.within_band(0.50)


class TestMetrics:
    def test_pinball_valores_conocidos(self):
        assert pinball_loss(0.5, 0.0, 0.02) == pytest.approx(0.01)
        assert pinball_loss(0.1, 0.01, -0.02) == pytest.approx(0.027)
        with pytest.raises(ValueError, match="level"):
            pinball_loss(1.5, 0.0, 0.0)

    def test_crps_normal_estandar_en_cero(self):
        assert crps_gaussian(0.0, 1.0, 0.0) == pytest.approx(0.2337, abs=1e-3)

    def test_crps_crece_con_error(self):
        assert crps_gaussian(0.0, 1.0, 3.0) > crps_gaussian(0.0, 1.0, 0.0)
        with pytest.raises(ValueError, match="std debe ser > 0"):
            crps_gaussian(0.0, 0.0, 0.0)


class TestPredictionIntegration:
    def test_sin_distribucion_backward_compatible(self):
        pred = _prediction()
        outcome = _outcome(pred, 0.002)
        assert evaluate_prediction(pred, outcome).distribution is None

    def test_con_distribucion_evalua_pinball_crps_cobertura(self):
        pred = _prediction(_dist())
        outcome = _outcome(pred, 0.002)
        dist_eval = evaluate_prediction(pred, outcome).distribution
        assert dist_eval is not None
        assert dist_eval.within_band is True
        assert dist_eval.tail_breach is False
        assert dist_eval.crps >= 0
        assert dist_eval.mean_pinball == pytest.approx(
            (dist_eval.pinball_10 + dist_eval.pinball_50 + dist_eval.pinball_90) / 3.0
        )

    def test_tail_breach_cuando_realizado_peor_que_cvar(self):
        pred = _prediction(_dist())
        outcome = _outcome(pred, -0.05)
        dist_eval = evaluate_prediction(pred, outcome).distribution
        assert dist_eval is not None
        assert dist_eval.tail_breach is True
        assert dist_eval.within_band is False

    def test_incoherencia_media_distribucion_falla(self):
        with pytest.raises(ValueError, match="distribución incoherente"):
            _prediction(_dist(), expected_return=0.99)

    def test_horizontes_distribucionales_1_5_15m(self):
        assert DISTRIBUTIONAL_HORIZONS == (60, 300, 900)
        for horizon in DISTRIBUTIONAL_HORIZONS:
            pred = _prediction(_dist(), horizon_seconds=horizon)
            assert pred.horizon_seconds == horizon


class TestQuantileCalibration:
    def test_shift_recupera_cobertura_nominal(self):
        realizeds = [i / 100.0 for i in range(10)]  # 0.00..0.09
        qvalues = [0.05] * 10  # cobertura 0.6, objetivo τ=0.1
        assert empirical_coverage(qvalues, realizeds) == pytest.approx(0.6)
        shift = fit_quantile_shift(0.10, qvalues, realizeds)
        assert shift.samples == 10
        shifted = [q + shift.shift for q in qvalues]
        assert empirical_coverage(shifted, realizeds) == pytest.approx(0.10)

    def test_sin_evidencia_shift_cero(self):
        shift = fit_quantile_shift(0.5, [0.01], [0.02])
        assert shift.shift == 0.0
        assert shift.samples == 1

    def test_apply_desplaza_cuantiles_e_inmutable(self):
        dist = _dist()
        shifted = apply_quantile_shifts(dist, {0.10: -0.002, 0.50: 0.0, 0.90: 0.002})
        assert shifted.quantile_10 == pytest.approx(dist.quantile_10 - 0.002)
        assert shifted.quantile_90 == pytest.approx(dist.quantile_90 + 0.002)
        assert shifted.expected_return == dist.expected_return
        assert dist.quantile_10 != shifted.quantile_10  # original intacto

    def test_shift_extremo_que_rompe_orden_falla(self):
        dist = _dist()
        with pytest.raises(ValueError, match="cuantiles desordenados"):
            apply_quantile_shifts(dist, {0.10: 0.50})  # q10 salta sobre q50

    def test_qvalues_realizeds_distinta_longitud_falla(self):
        with pytest.raises(ValueError, match="igual longitud"):
            fit_quantile_shift(0.5, [0.01, 0.02], [0.01])
        with pytest.raises(ValueError, match="igual longitud"):
            empirical_coverage([0.01], [0.01, 0.02])

    def test_nivel_invalido_falla(self):
        with pytest.raises(ValueError, match="level"):
            fit_quantile_shift(0.0, [], [])
        with pytest.raises(ValueError, match="level"):
            pinball_loss(0.0, 0.0, 0.0)
