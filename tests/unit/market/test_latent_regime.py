"""FASE 2 — Régimen latente: estados no etiquetados del mercado.

TRENDING / MEAN_REVERTING / LOW_VOL / HIGH_VOL / BREAKOUT / CRASH
inferidos por prototipos + filtro temporal + CUSUM. El clasificador
determinista sigue intacto (baseline); lo latente vive al lado.
"""

from __future__ import annotations

import math

import pytest
from iot_machine_learning.domain.entities.market import Candle, DataStatus
from iot_machine_learning.domain.entities.market.calibration import ContextKey
from iot_machine_learning.domain.entities.market.prediction import Regime
from iot_machine_learning.domain.entities.market.regime import (
    LatentRegime,
    RegimeFilter,
    RegimePosterior,
    TransitionMatrix,
    cusum_changepoint,
    regime_features,
    score_posterior,
    uniform_posterior,
)
from iot_machine_learning.domain.entities.market.replay import FeatureWindow
from iot_machine_learning.domain.entities.market.replay.classifier import (
    LATENT_TO_REGIME,
    classify_latent_regime,
    classify_regime,
    latent_posterior,
)


def _window_from_returns(returns: list[float], symbol: str = "X") -> FeatureWindow:
    """Ventana con N+1 velas para N retornos (los retornos log = la serie)."""
    window = FeatureWindow(symbol=symbol)
    prev = 100.0
    window = window.append_closed(Candle(
        symbol=symbol, timestamp=60.0,
        data_status=DataStatus.REPLAY, source_provider="latent-test",
        open=prev, high=prev * 1.0001, low=prev * 0.9999, close=prev,
        volume=100.0, interval_seconds=60,
    ))
    for i, r in enumerate(returns):
        close = prev * math.exp(r)
        hi = max(prev, close) * 1.0001
        lo = min(prev, close) * 0.9999
        window = window.append_closed(Candle(
            symbol=symbol, timestamp=float(120 + i * 60),
            data_status=DataStatus.REPLAY, source_provider="latent-test",
            open=prev, high=hi, low=lo, close=close,
            volume=100.0, interval_seconds=60,
        ))
        prev = close
    return window


N = 20
ALTERN = [0.01 if i % 2 == 0 else -0.01 for i in range(N)]
PPNN = ([0.02, 0.02, -0.02, -0.02] * 5)[:N]
DRIFT = [0.002 + 0.001 * math.sin(i) for i in range(N)]
FLAT = [0.0001 + 0.0002 * math.sin(i) for i in range(N)]
BREAKOUT = [0.0005 + 0.0003 * math.sin(i) for i in range(N - 1)] + [0.03]
CRASH = [-0.002 + 0.003 * math.sin(i) for i in range(N)]
CRASH[7] = -0.035
CRASH[14] = -0.028


class TestLatentStates:
    @pytest.mark.parametrize(("returns", "expected"), [
        (ALTERN, LatentRegime.MEAN_REVERTING),
        (PPNN, LatentRegime.HIGH_VOL),
        (DRIFT, LatentRegime.TRENDING),
        (FLAT, LatentRegime.LOW_VOL),
        (BREAKOUT, LatentRegime.BREAKOUT),
        (CRASH, LatentRegime.CRASH),
    ])
    def test_map_por_serie_sintetica(self, returns, expected):
        posterior = score_posterior(regime_features(returns))
        assert posterior.most_likely is expected
        assert posterior.confidence > 0.5

    def test_posterior_suma_uno_y_determinista(self):
        first = score_posterior(regime_features(DRIFT))
        second = score_posterior(regime_features(DRIFT))
        assert sum(first.probs) == pytest.approx(1.0)
        assert first.probs == second.probs
        assert first.entropy < math.log(6)

    def test_uniforme_maxima_entropia(self):
        assert uniform_posterior().entropy == pytest.approx(math.log(6))

    def test_probs_invalidas_fallan(self):
        with pytest.raises(ValueError, match="sumar 1.0"):
            RegimePosterior(probs=(0.5,) * 6)
        with pytest.raises(ValueError, match="6 elementos"):
            RegimePosterior(probs=(0.5, 0.5))

    def test_retornos_insuficientes_fallan(self):
        with pytest.raises(ValueError, match="insuficientes"):
            regime_features([0.01] * 5)


class TestWindowIntegration:
    def test_ventana_honesta_solo_velas_cerradas(self):
        window = _window_from_returns(DRIFT)
        posterior = latent_posterior(window)
        assert posterior is not None
        assert posterior.most_likely is LatentRegime.TRENDING
        # Walk-forward honesto: el prefijo usa solo pasado ya ocurrido.
        prefix = FeatureWindow(symbol="X", candles=window.candles[:15])
        first = latent_posterior(prefix, lookback=10)
        assert first is not None
        assert first.probs == latent_posterior(prefix, lookback=10).probs
        # Y coincide con el cálculo directo sobre esos retornos.
        direct = score_posterior(regime_features(prefix.returns(10)))
        assert first.probs == direct.probs

    def test_ventana_insuficiente_retorna_none(self):
        window = _window_from_returns([0.001] * 5)
        assert latent_posterior(window) is None
        assert classify_latent_regime(window) is None
    def test_determinista_sigue_vivo(self):
        window = _window_from_returns(DRIFT)
        assert classify_regime(window) in (
            Regime.BULL, Regime.BEAR, Regime.NEUTRAL, Regime.HIGH_VOLATILITY,
        )

    def test_mapeo_latente_a_regime_cubre_los_seis(self):
        assert set(LATENT_TO_REGIME) == set(LatentRegime)
        window = _window_from_returns(CRASH)
        assert classify_latent_regime(window) is Regime.BEAR

    def test_posterior_fluye_a_context_key(self):
        posterior = latent_posterior(_window_from_returns(DRIFT))
        assert posterior is not None
        key = ContextKey(
            strategy="ml-v1", horizon_seconds=300,
            regime=posterior.most_likely.value,
        )
        assert "TRENDING" in str(key)


class TestTransitionMatrix:
    def test_filas_suman_uno_incluso_sin_datos(self):
        matrix = TransitionMatrix()
        for prev in LatentRegime:
            total = sum(matrix.prob(prev, curr) for curr in LatentRegime)
            assert total == pytest.approx(1.0)
            assert matrix.prob(prev, prev) == pytest.approx(1.0 / 6)

    def test_conteo_aprende_persistencia(self):
        matrix = TransitionMatrix()
        for _ in range(9):
            matrix.observe(LatentRegime.TRENDING, LatentRegime.TRENDING)
        matrix.observe(LatentRegime.TRENDING, LatentRegime.CRASH)
        assert matrix.prob(LatentRegime.TRENDING, LatentRegime.TRENDING) > 0.5
        assert matrix.transitions_observed == 10


class TestRegimeFilter:
    def test_persistencia_eleva_confianza(self):
        filt = RegimeFilter()
        likelihood = score_posterior(regime_features(DRIFT))
        confidences = [filt.update(likelihood).confidence for _ in range(4)]
        assert confidences[-1] > confidences[0]
        assert filt.stable_regime is LatentRegime.TRENDING
        assert filt.transitions.transitions_observed == 3

    def test_sin_evidencia_no_hay_regimen_estable(self):
        assert RegimeFilter().stable_regime is None

    def test_likelihood_invalida_falla(self):
        with pytest.raises(TypeError, match="RegimePosterior"):
            RegimeFilter().update("TRENDING")


class TestChangepoint:
    def test_shift_sostenido_se_detecta(self):
        returns = [0.001 * math.sin(i) for i in range(30)]
        returns += [0.008 + 0.001 * math.sin(i) for i in range(30)]
        hit = cusum_changepoint(tuple(returns))
        assert hit is not None
        assert hit >= 25  # no dispara antes del cambio real

    def test_serie_estable_sin_disparo(self):
        returns = [0.001 * math.sin(i) for i in range(40)]
        assert cusum_changepoint(tuple(returns)) is None

    def test_insuficiente_falla(self):
        with pytest.raises(ValueError, match="insuficientes"):
            cusum_changepoint((0.01, 0.02))
