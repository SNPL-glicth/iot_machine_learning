"""FASE 7 — Aprender de los propios errores (taxonomía + ledger).

Prediction → outcome → error → ¿qué falló? → directiva → ledger
(propone; el refit offline aplica). Sin aprendizaje en vivo.
"""

from __future__ import annotations

import json

import pytest
from iot_machine_learning.domain.entities.market import DataStatus, Quote
from iot_machine_learning.domain.entities.market.costs import COST_PROFILES
from iot_machine_learning.domain.entities.market.learning import (
    LEDGER_STATE_VERSION,
    AdaptationAction,
    ErrorCause,
    LearningLedger,
    attribute_error,
    attribute_prediction,
    propose_directive,
)
from iot_machine_learning.domain.entities.market.prediction import (
    Outcome,
    Prediction,
    PredictionInterval,
    Regime,
    ReturnDistribution,
    evaluate_prediction,
)

BTC = COST_PROFILES["BTC-USD"]
TS = 1_600_000_000.0


def _attribution(**overrides):
    kwargs = dict(
        direction_correct=False,
        magnitude_error=0.01,
        calibration_error=0.2,
        confidence=0.6,
        net_return=0.001,
    )
    kwargs.update(overrides)
    return attribute_error(**kwargs)


class TestTaxonomy:
    def test_quiet_cuando_todo_bien(self):
        attribution = _attribution(
            direction_correct=True, magnitude_error=0.001)
        assert attribution.primary is ErrorCause.QUIET
        assert attribution.contributors == (ErrorCause.QUIET,)

    def test_signal_decay_por_defecto(self):
        attribution = _attribution()
        assert attribution.primary is ErrorCause.SIGNAL_DECAY

    def test_tail_gana_a_todo(self):
        attribution = _attribution(
            tail_breach=True, regime_at_predict="TRENDING",
            regime_at_resolve="CRASH", data_degraded=True, confidence=0.95)
        assert attribution.primary is ErrorCause.TAIL_EVENT
        assert ErrorCause.REGIME_SHIFT in attribution.contributors
        assert ErrorCause.DATA_DEGRADED in attribution.contributors

    def test_regime_shift_antes_que_sobreconfianza(self):
        attribution = _attribution(
            regime_at_predict="LOW_VOL", regime_at_resolve="CRASH",
            confidence=0.95)
        assert attribution.primary is ErrorCause.REGIME_SHIFT
        assert ErrorCause.OVERCONFIDENCE in attribution.contributors

    def test_sin_cambio_de_regimen_no_hay_shift(self):
        attribution = _attribution(
            regime_at_predict="TRENDING", regime_at_resolve="TRENDING",
            confidence=0.95)
        assert attribution.primary is ErrorCause.OVERCONFIDENCE
        assert ErrorCause.REGIME_SHIFT not in attribution.contributors

    def test_dato_degradado_con_error_material(self):
        attribution = _attribution(data_degraded=True)
        assert attribution.primary is ErrorCause.DATA_DEGRADED

    def test_dato_degradado_con_error_tiny_es_quiet(self):
        attribution = _attribution(
            direction_correct=True, magnitude_error=0.001,
            data_degraded=True)
        assert attribution.primary is ErrorCause.QUIET

    def test_cost_kill_direccion_buena_neto_malo(self):
        attribution = _attribution(
            direction_correct=True, magnitude_error=0.002, net_return=-0.001)
        assert attribution.primary is ErrorCause.COST_KILL

    def test_sobreconfianza(self):
        attribution = _attribution(confidence=0.9)
        assert attribution.primary is ErrorCause.OVERCONFIDENCE

    def test_deriva_de_calibracion(self):
        attribution = _attribution(
            direction_correct=True, magnitude_error=0.002,
            calibration_error=0.6, net_return=0.002)
        assert attribution.primary is ErrorCause.CALIBRATION_DRIFT

    def test_entradas_invalidas_fallan(self):
        with pytest.raises(ValueError, match="magnitude_error"):
            _attribution(magnitude_error=-1.0)
        with pytest.raises(ValueError, match="calibration_error"):
            _attribution(calibration_error=1.5)
        with pytest.raises(ValueError, match="confidence"):
            _attribution(confidence=2.0)
        with pytest.raises(ValueError, match="net_return"):
            _attribution(net_return=float("nan"))

    def test_detail_trazable(self):
        attribution = _attribution(
            regime_at_predict="A", regime_at_resolve="A")
        assert "signal_decay" in attribution.detail
        assert "A→A" in attribution.detail


class TestDirectives:
    def _directive(self, cause: str, **overrides):
        attribution = _attribution(**overrides)
        assert attribution.primary.value == cause
        return propose_directive(
            attribution, expert="momentum", regime="TRENDING",
            horizon_seconds=300, prediction_id="p-1")

    def test_mapa_causa_accion(self):
        assert self._directive("signal_decay").action is AdaptationAction.DOWNWEIGHT_EXPERT
        assert self._directive("quiet", direction_correct=True,
                               magnitude_error=0.001).action is AdaptationAction.NO_ACTION
        assert self._directive("overconfidence",
                               confidence=0.9).action is AdaptationAction.RECALIBRATE
        assert self._directive("tail_event",
                               tail_breach=True).action is AdaptationAction.WIDEN_TAILS
        assert self._directive("regime_shift", regime_at_predict="A",
                               regime_at_resolve="B").action is AdaptationAction.REFIT_REGIME
        assert self._directive("data_degraded",
                               data_degraded=True).action is AdaptationAction.QUARANTINE_FEED
        assert self._directive(
            "cost_kill", direction_correct=True, magnitude_error=0.002,
            net_return=-0.001).action is AdaptationAction.REVIEW_COSTS
        assert self._directive(
            "calibration_drift", direction_correct=True,
            magnitude_error=0.002, calibration_error=0.6,
            net_return=0.002).action is AdaptationAction.RECALIBRATE

    def test_target_y_razon_trazables(self):
        directive = self._directive("signal_decay")
        assert directive.target == "momentum|TRENDING|300s"
        assert "signal_decay" in directive.reason
        assert directive.cause is ErrorCause.SIGNAL_DECAY

    def test_directiva_huerfana_falla(self):
        attribution = _attribution()
        directive = propose_directive(
            attribution, expert="momentum", regime="X",
            horizon_seconds=60, prediction_id="p-2")
        other = _attribution(direction_correct=True, magnitude_error=0.001)
        assert directive.cause is not other.primary
        with pytest.raises(TypeError, match="ErrorAttribution"):
            propose_directive("nope", expert="m", regime="X",
                              horizon_seconds=60, prediction_id="p-3")


class TestLedger:
    def _proposed(self, ledger: LearningLedger, prediction_id: str = "p-1"):
        attribution = _attribution()
        directive = propose_directive(
            attribution, expert="momentum", regime="TRENDING",
            horizon_seconds=300, prediction_id=prediction_id)
        return ledger.propose(attribution, directive)

    def test_propose_y_pending(self):
        ledger = LearningLedger()
        record = self._proposed(ledger)
        assert record.record_id == "learn-000001"
        assert record.applied is False
        assert len(ledger.pending()) == 1

    def test_mark_applied(self):
        ledger = LearningLedger()
        record = self._proposed(ledger)
        updated = ledger.mark_applied(record.record_id, "refit-v7")
        assert updated.applied is True
        assert updated.applied_version == "refit-v7"
        assert ledger.pending() == ()
        assert ledger.summary()["applied"] == 1

    def test_summary_por_causa(self):
        ledger = LearningLedger()
        self._proposed(ledger, "p-1")
        self._proposed(ledger, "p-2")
        summary = ledger.summary()
        assert summary["by_cause"] == {"signal_decay": 2}
        assert summary["by_action"] == {"downweight_expert": 2}
        assert summary["total"] == 2 and summary["pending"] == 2

    def test_roundtrip_json(self):
        ledger = LearningLedger()
        record = self._proposed(ledger)
        ledger.mark_applied(record.record_id, "refit-v7")
        restored = LearningLedger.from_state(
            json.loads(json.dumps(ledger.to_state())))
        assert restored.summary() == ledger.summary()
        assert restored.pending() == ()

    def test_version_desconocida_falla(self):
        with pytest.raises(ValueError, match="versión"):
            LearningLedger.from_state({"version": "ovni-v1"})

    def test_record_desconocido_falla(self):
        with pytest.raises(KeyError, match="desconocido"):
            LearningLedger().mark_applied("learn-999999", "v1")

    def test_propuesta_incoherente_falla(self):
        ledger = LearningLedger()
        attribution = _attribution()
        directive = propose_directive(
            attribution, expert="m", regime="X",
            horizon_seconds=60, prediction_id="p-9")
        other = _attribution(direction_correct=True, magnitude_error=0.001)
        with pytest.raises(ValueError, match="primary"):
            ledger.propose(other, directive)


class TestEndToEnd:
    def _prediction(self, expected: float, confidence: float) -> Prediction:
        quote = Quote(
            symbol="BTC-USD", timestamp=TS, data_status=DataStatus.REALTIME,
            source_provider="binance", bid=50000.0, bid_size=1.0,
            ask=50010.0, ask_size=1.0)
        dist = ReturnDistribution(
            expected_return=expected, volatility=0.01,
            quantile_10=expected - 0.0128, quantile_50=expected,
            quantile_90=expected + 0.0128, cvar_05=-0.02)
        return Prediction(
            prediction_id="btc-e2e", observation=quote, horizon_seconds=300,
            timestamp=TS, entry_price=quote.midpoint,
            expected_return=expected, probability_up=0.7,
            confidence=confidence, distribution=dist,
            regime=Regime.BULL, strategy="momentum")

    def _outcome(self, pred: Prediction, realized: float) -> Outcome:
        return Outcome(
            symbol="BTC-USD", observation_timestamp=TS,
            horizon_seconds=300, measured_at=TS + 300,
            final_price=pred.entry_price * (1.0 + realized),
            return_realized=realized)

    def test_cola_real_atribuye_tail_event(self):
        pred = self._prediction(0.0018, 0.6)
        outcome = self._outcome(pred, -0.05)  # peor que cvar_05
        evaluation = evaluate_prediction(
            pred.activate().to_waiting_outcome(outcome), outcome)
        assert evaluation.distribution is not None
        assert evaluation.distribution.tail_breach is True
        attribution = attribute_prediction(
            pred, outcome, evaluation, regime_at_resolve="CRASH",
            cost_model=BTC)
        assert attribution.primary is ErrorCause.TAIL_EVENT

    def test_fallo_sobreconfiado_propone_recalibrar(self):
        pred = self._prediction(0.0018, 0.95)
        outcome = self._outcome(pred, -0.004)
        evaluation = evaluate_prediction(
            pred.activate().to_waiting_outcome(outcome), outcome)
        ledger = LearningLedger()
        attribution = attribute_prediction(
            pred, outcome, evaluation, regime_at_resolve="bull",
            cost_model=BTC)
        assert attribution.primary is ErrorCause.OVERCONFIDENCE
        directive = propose_directive(
            attribution, expert="momentum", regime="bull",
            horizon_seconds=300, prediction_id="btc-e2e")
        record = ledger.propose(attribution, directive)
        assert record.action is AdaptationAction.RECALIBRATE
        assert ledger.summary()["by_cause"] == {"overconfidence": 1}
