"""FASE 3 — Microestructura L1: el tape precede al movimiento.

Spread, imbalance, OFI, agresión buy/sell, intensidad, impacto +
predictor short-horizon que emite Prediction con distribución
al ciclo Outcome→Evaluation→Reward.
"""

from __future__ import annotations

import pytest
from iot_machine_learning.domain.entities.market import (
    Capability,
    DataStatus,
    ProviderProfile,
    Quote,
    Trade,
)
from iot_machine_learning.domain.entities.market.audit import audit_feed
from iot_machine_learning.domain.entities.market.microstructure import (
    MICRO_HORIZONS,
    MICRO_STRATEGY,
    MicroWindow,
    l1_available,
    l1_features,
    predict_micro,
    to_prediction,
)
from iot_machine_learning.domain.entities.market.prediction import (
    Outcome,
    evaluate_prediction,
)


def _q(ts: float, bid: float, bs: float, ask: float, az: float,
       symbol: str = "BTC-USD") -> Quote:
    return Quote(
        symbol=symbol, timestamp=ts, data_status=DataStatus.REALTIME,
        source_provider="binance", bid=bid, bid_size=bs, ask=ask, ask_size=az,
    )


def _t(ts: float, px: float, sz: float, side: str | None) -> Trade:
    return Trade(
        symbol="BTC-USD", timestamp=ts, data_status=DataStatus.REALTIME,
        source_provider="binance", price=px, size=sz, taker_side=side,
    )


BID_HEAVY = (
    _q(100.0, 99.0, 10.0, 101.0, 5.0),
    _q(101.0, 99.0, 15.0, 101.0, 5.0),
)
BUYS = (
    _t(100.2, 100.0, 2.0, "buy"),
    _t(100.5, 100.0, 1.0, "sell"),
)


class TestL1Features:
    def test_valores_a_mano(self):
        f = l1_features(BID_HEAVY, BUYS)
        assert f.imbalance == pytest.approx(0.5)  # (15-5)/20
        assert f.signed_volume_ratio == pytest.approx(1 / 3)
        assert f.ofi == pytest.approx(5 / 35)  # +5 size / depth 35
        assert f.spread_bps == pytest.approx(200.0)
        assert f.mid_drift == pytest.approx(0.0)
        assert f.trade_intensity == pytest.approx(2.0)
        assert f.crossed_pct == pytest.approx(0.0)
        assert f.buy_volume == pytest.approx(2.0)
        assert f.sell_volume == pytest.approx(1.0)

    def test_ofi_bid_up_paga_size_nuevo(self):
        quotes = (_q(100.0, 99.0, 10.0, 101.0, 5.0),
                  _q(101.0, 99.5, 7.0, 101.0, 5.0))
        f = l1_features(quotes)
        # bid up: +7; ask same: -(5-5)=0; depth=(10+5)+(7+5)=27
        assert f.ofi == pytest.approx(7 / 27)

    def test_ofi_ask_down_presiona_venta(self):
        quotes = (_q(100.0, 99.0, 10.0, 101.0, 5.0),
                  _q(101.0, 99.0, 10.0, 100.5, 8.0))
        f = l1_features(quotes)
        # bid same: 0; ask down: -8; depth=(10+5)+(10+8)=33
        assert f.ofi == pytest.approx(-8 / 33)

    def test_libro_cruzado_se_mide_no_se_rechaza(self):
        quotes = (_q(100.0, 101.0, 1.0, 100.0, 1.0),
                  _q(101.0, 101.0, 1.0, 100.0, 1.0))
        f = l1_features(quotes)
        assert f.crossed_pct == pytest.approx(1.0)
        assert f.spread_bps < 0

    def test_sin_trades_agresion_cero(self):
        f = l1_features(BID_HEAVY)
        assert f.signed_volume_ratio == 0.0
        assert f.price_impact == 0.0
        assert f.n_trades == 0

    def test_quotes_insuficientes_fallan(self):
        with pytest.raises(ValueError, match="insuficientes"):
            l1_features((BID_HEAVY[0],))

    def test_mezcla_simbolos_falla(self):
        other = _q(102.0, 99.0, 1.0, 101.0, 1.0, symbol="SPY")
        with pytest.raises(ValueError, match="mezclan símbolos"):
            l1_features((BID_HEAVY[0], other))


class TestMicroWindow:
    def test_append_ordena_y_rechaza_duplicados(self):
        window = MicroWindow("BTC-USD")
        window = window.append_quote(BID_HEAVY[0]).append_quote(BID_HEAVY[1])
        assert window.size == 2
        with pytest.raises(ValueError, match="fuera de orden o duplicado"):
            window.append_quote(BID_HEAVY[1])
        with pytest.raises(ValueError, match="otro símbolo"):
            window.append_quote(_q(102.0, 99.0, 1.0, 101.0, 1.0, symbol="SPY"))

    def test_features_y_puente_auditoria(self):
        window = MicroWindow("BTC-USD", BID_HEAVY, BUYS)
        assert window.features().imbalance == pytest.approx(0.5)
        events = window.audited()
        assert len(events) == 4
        report = audit_feed(list(events), symbol="BTC-USD",
                            expected_interval_seconds=1.0)
        assert report.duplicates == 0
        assert report.total == 4

    def test_duplicado_en_tape_lo_ve_la_fase_0(self):
        window = MicroWindow("BTC-USD", BID_HEAVY + (BID_HEAVY[0],), BUYS)
        report = audit_feed(list(window.audited()), symbol="BTC-USD",
                            expected_interval_seconds=1.0)
        assert report.duplicates == 1


class TestCapabilityGate:
    def _profile(self, *caps: Capability) -> ProviderProfile:
        return ProviderProfile(
            provider="test", asset_class="test",
            capabilities=frozenset(caps),
        )

    def test_binance_y_alpaca_tienen_l1(self):
        from iot_machine_learning.infrastructure.adapters.market.binance.constants import (
            BINANCE_PROFILE,
        )
        from iot_machine_learning.infrastructure.adapters.market.alpaca.constants import (
            ALPACA_PROFILE,
        )
        assert l1_available(BINANCE_PROFILE) is True
        assert l1_available(ALPACA_PROFILE) is True

    def test_sin_quotes_no_hay_l1(self):
        assert l1_available(self._profile(Capability.TRADES)) is False
        assert l1_available(self._profile(
            Capability.TRADES, Capability.QUOTES)) is True


class TestMicroPredictor:
    def test_bid_heavy_con_compras_mira_arriba(self):
        signal = predict_micro(l1_features(BID_HEAVY, BUYS))
        assert signal.probability_up == pytest.approx(0.6930, abs=1e-3)
        assert signal.expected_return > 0
        assert signal.score == pytest.approx(0.2714285, abs=1e-6)

    def test_escenario_espejo_mira_abajo(self):
        mirror_q = (
            _q(100.0, 99.0, 5.0, 101.0, 10.0),
            _q(101.0, 99.0, 5.0, 101.0, 15.0),
        )
        mirror_t = (
            _t(100.2, 100.0, 1.0, "buy"),
            _t(100.5, 100.0, 2.0, "sell"),
        )
        signal = predict_micro(l1_features(mirror_q, mirror_t))
        assert signal.probability_up < 0.5
        assert signal.expected_return < 0

    def test_determinista(self):
        features = l1_features(BID_HEAVY, BUYS)
        assert predict_micro(features) == predict_micro(features)

    def test_sin_trades_tambien_predice(self):
        signal = predict_micro(l1_features(BID_HEAVY))
        assert 0.05 <= signal.probability_up <= 0.95
        assert signal.volatility > 0

    def test_horizontes_micro(self):
        assert MICRO_HORIZONS == (30, 60)
        assert MICRO_STRATEGY == "micro-l1"


class TestToPrediction:
    def test_prediccion_lista_para_el_ciclo(self):
        signal = predict_micro(l1_features(BID_HEAVY, BUYS))
        pred = to_prediction(
            signal, anchor=BID_HEAVY[-1], prediction_id="btc-micro-1",
            horizon_seconds=60, timestamp=101.0,
        )
        assert pred.entry_price == pytest.approx(100.0)  # midpoint
        assert pred.strategy == "micro-l1"
        assert pred.distribution is not None
        assert pred.distribution.expected_return == pred.expected_return
        assert pred.observation is BID_HEAVY[-1]

    def test_ciclo_completo_hasta_evaluacion(self):
        signal = predict_micro(l1_features(BID_HEAVY, BUYS))
        pred = to_prediction(
            signal, anchor=BID_HEAVY[-1], prediction_id="btc-micro-2",
            horizon_seconds=60, timestamp=101.0,
        )
        outcome = Outcome(
            symbol="BTC-USD", observation_timestamp=101.0,
            horizon_seconds=60, measured_at=161.0,
            final_price=100.3, return_realized=0.003,
        )
        evaluation = evaluate_prediction(
            pred.activate().to_waiting_outcome(outcome), outcome,
        )
        assert evaluation.distribution is not None
        assert evaluation.direction_correct is True
