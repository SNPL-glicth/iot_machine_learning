"""FASE 10.5 — Tests de integración calibrador ↔ pipeline de emisión."""

from __future__ import annotations

import random

import pytest

from iot_machine_learning.domain.entities.market.calibration import (
    AdaptiveCalibrator,
    CalibrationMethod,
    ContextKey,
)
from iot_machine_learning.domain.entities.market.calibration.pipeline import (
    CalibratedPredictor,
    collect_training_pairs,
    try_refit,
    wrap_predictor,
)
from iot_machine_learning.domain.entities.market.observations import Candle
from iot_machine_learning.domain.entities.market.replay.feature_window import (
    FeatureWindow,
)
from iot_machine_learning.domain.entities.market import DataStatus
from iot_machine_learning.domain.entities.market.prediction.types import Regime
from iot_machine_learning.domain.entities.market.prediction.evaluation import (
    Evaluation,
)


# ─── Fábricas ───────────────────────────────────────────────────────────────


class OverconfidentPredictor:
    """Predictor crudo descalibrado: siempre P(up)=0.90, retorno fijo."""

    def __init__(self, name: str = "momentum") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def predict(
        self, window, *, horizon_seconds, observation_interval, lookback
    ):
        from iot_machine_learning.domain.entities.market.replay.baselines import (
            PredictionSignal,
        )

        return PredictionSignal(
            probability_up=0.90,
            expected_return=0.001,
            lower=-0.01,
            upper=0.02,
        )


def _candle(ts: int, close: float) -> Candle:
    return Candle(
        symbol="BTC-USD",
        timestamp=float(ts),
        data_status=DataStatus.REPLAY,
        source_provider="test",
        interval_seconds=60,
        open=close,
        high=close * 1.001,
        low=close * 0.999,
        close=close,
        volume=1.0,
    )


def _window() -> FeatureWindow:
    window = FeatureWindow(symbol="BTC-USD")
    for i in range(30):
        window = window.append_closed(_candle(i * 60 + 60, 100.0 + (i % 5)))
    return window


def _prediction(strategy="momentum", prob=0.9, hit=True, horizon=900):
    """Stub con el subconjunto del contrato que usa collect_training_pairs."""

    class _Pred:
        pass

    p = _Pred()
    p.strategy = strategy
    p.horizon_seconds = horizon
    p.regime = Regime.NEUTRAL if strategy == "momentum" else None
    p.probability_up = prob
    p.evaluation = Evaluation(
        direction_correct=hit,
        magnitude_error=0.01,
        within_interval=True,
        calibration_error=abs(prob - float(hit)),
    )
    return p


# ─── Passthrough honesto ────────────────────────────────────────────────────


class TestPassthrough:
    def test_unfitted_calibrator_emite_probabilidad_cruda(self):
        wrapper = wrap_predictor(OverconfidentPredictor(), AdaptiveCalibrator())
        signal = wrapper.predict(
            _window(), horizon_seconds=900, observation_interval=60, lookback=20
        )
        assert signal.probability_up == 0.90
        # No-negociable 3: sin evidencia queda marcado, nunca inventado.
        assert wrapper.last_fallback_level == "UNCALIBRATED"
        evidence = wrapper.evidence_log[-1]
        assert evidence.prob_raw == evidence.prob_calibrated == 0.90
        assert evidence.fallback_level == "UNCALIBRATED"

    def test_expected_return_e_intervalo_intactos(self):
        raw = OverconfidentPredictor().predict(
            _window(), horizon_seconds=900, observation_interval=60, lookback=20
        )
        wrapper = wrap_predictor(OverconfidentPredictor(), AdaptiveCalibrator())
        calibrated = wrapper.predict(
            _window(), horizon_seconds=900, observation_interval=60, lookback=20
        )
        assert calibrated.expected_return == raw.expected_return
        assert calibrated.lower == raw.lower
        assert calibrated.upper == raw.upper

    def test_name_con_y_sin_sufijo(self):
        inner = OverconfidentPredictor()
        assert wrap_predictor(inner, AdaptiveCalibrator()).name == "momentum"
        suffixed = wrap_predictor(inner, AdaptiveCalibrator(), name_suffix="-cal")
        assert suffixed.name == "momentum-cal"
        # La clave de contexto SIEMPRE usa el nombre crudo.
        assert suffixed.inner_name == "momentum"


# ─── Corrección real ────────────────────────────────────────────────────────


class TestCorrection:
    def _fitted_wrapper(self, n=600, seed=7) -> CalibratedPredictor:
        rng = random.Random(seed)
        context = ContextKey("momentum", 900, "ALL")
        pairs = [
            (context, 0.90, rng.random() < 0.50) for _ in range(n)
        ]
        calibrator = AdaptiveCalibrator(method=CalibrationMethod.PLATT)
        accepted = try_refit(calibrator, pairs)
        assert accepted is True
        return wrap_predictor(OverconfidentPredictor(), calibrator)

    def test_p09_sobre_moneda_se_corrige_hacia_05_07(self):
        wrapper = self._fitted_wrapper()
        signal = wrapper.predict(
            _window(), horizon_seconds=900, observation_interval=60, lookback=20
        )
        assert 0.45 <= signal.probability_up <= 0.70
        assert wrapper.last_fallback_level is not None

    def test_probabilidad_dentro_del_contrato_005_095(self):
        wrapper = self._fitted_wrapper()
        for hz in (300, 900, 1800):
            signal = wrapper.predict(
                _window(),
                horizon_seconds=hz,
                observation_interval=60,
                lookback=20,
            )
            assert 0.05 <= signal.probability_up <= 0.95

    def test_refit_sin_evidencia_suficiente_rechaza_y_pasa_crudo(self):
        # Debajo de min_train+val+test (100+50+50): train_and_evaluate
        # devuelve None ⇒ try_refit False ⇒ el wrapper queda en passthrough.
        context = ContextKey("momentum", 900, "ALL")
        pairs = [(context, 0.9, i % 2 == 0) for i in range(90)]
        calibrator = AdaptiveCalibrator(method=CalibrationMethod.PLATT)
        assert try_refit(calibrator, pairs) is False
        wrapper = wrap_predictor(OverconfidentPredictor(), calibrator)
        signal = wrapper.predict(
            _window(), horizon_seconds=900, observation_interval=60, lookback=20
        )
        assert signal.probability_up == 0.90
        assert wrapper.last_fallback_level == "UNCALIBRATED"


# ─── Dataset builder ────────────────────────────────────────────────────────


class TestCollectTrainingPairs:
    def test_filtra_sin_evaluacion_y_mapea_contexto(self):
        resolved = [
            _prediction(strategy="momentum", prob=0.9, hit=True),
            _prediction(strategy="ema", prob=0.3, hit=False),
            _prediction(strategy="naive", prob=0.6, hit=True),
        ]
        resolved[2].evaluation = None
        pairs = collect_training_pairs(resolved)

        assert len(pairs) == 2
        (ctx_m, prob_m, hit_m), (ctx_e, prob_e, hit_e) = pairs
        assert ctx_m.strategy == "momentum"
        assert ctx_m.horizon_seconds == 900
        assert ctx_m.regime == "neutral"
        assert (prob_m, hit_m) == (0.9, True)
        assert ctx_e.regime == "ALL"
        assert (prob_e, hit_e) == (0.3, False)

    def test_strategy_none_caen_en_baseline(self):
        pred = _prediction(strategy=None)
        (ctx, prob, _hit), = collect_training_pairs([pred])
        assert ctx.strategy == "baseline"


# ─── Fallback explícito (no-negociable 3) ───────────────────────────────────


class TestFallbackHierarchy:
    def _cal_sin_contexto_fino(self):
        """12 regímenes × 24 muestras por horizonte: celdas finas bajo el
        mínimo (CONTEXT nunca se construye) pero horizontes con evidencia
        sobrada (HORIZON se construye y alcanza val/test mínimos)."""
        rng = random.Random(11)
        pairs = []
        for hz in (900, 1800):
            for r in range(12):
                cell = [
                    (
                        ContextKey("momentum", hz, f"r{r}"),
                        0.90,
                        rng.random() < 0.45,
                    )
                    for _ in range(24)
                ]
                pairs.extend(cell)
        calibrator = AdaptiveCalibrator(method=CalibrationMethod.PLATT)
        accepted, comps = None, None
        calibrator2 = calibrator
        cals, comps = calibrator.train_and_evaluate(pairs)
        assert cals is not None
        return wrap_predictor(OverconfidentPredictor(), calibrator)

    def test_cae_en_horizon_cuando_contexto_no_alcanza(self):
        wrapper = self._cal_sin_contexto_fino()
        signal = wrapper.predict(
            _window(), horizon_seconds=900, observation_interval=60, lookback=20
        )
        # La ventana plana clasifica como ALL; el contexto fino no existe
        # en ningún calibrador ⇒ nivel HORIZON (strategy·horizon).
        assert wrapper.last_fallback_level == "horizon"
        assert signal.probability_up != 0.90

    def test_orden_jerarquia_contexto_horizon_estrategia_global(self):
        from iot_machine_learning.domain.entities.market.calibration.fallback import (
            get_fallback_calibrator,
            resolve_fallback_context,
        )
        from iot_machine_learning.domain.entities.market.calibration.verdicts import (
            FallbackLevel,
        )

        ctx = ContextKey("momentum", 900, "ALL")
        # Solo GLOBAL disponible ⇒ selección y clave canónica correctas.
        global_cal = ContextCalibratedStub("GLOBAL")
        level, cal = get_fallback_calibrator(ctx, {FallbackLevel.GLOBAL: global_cal})
        assert level == FallbackLevel.GLOBAL
        assert resolve_fallback_context(ctx, level) == ContextKey("GLOBAL", 0, "ALL")

    def test_uncalibrated_no_inventa_confianza(self):
        cal_vacio = AdaptiveCalibrator()
        wrapper = wrap_predictor(OverconfidentPredictor(), cal_vacio)
        raw = OverconfidentPredictor().predict(
            _window(), horizon_seconds=900, observation_interval=60, lookback=20
        )
        out = wrapper.predict(
            _window(), horizon_seconds=900, observation_interval=60, lookback=20
        )
        assert out.probability_up == raw.probability_up
        assert wrapper.last_fallback_level == "UNCALIBRATED"


class ContextCalibratedStub:
    """Sustituto mínimo de ContextCalibrator para pruebas de lookup."""

    def __init__(self, key_strategy: str) -> None:
        self._key = key_strategy

    @property
    def _params(self):  # noqa: SLF001 - el lookup accede aquí por diseño
        from iot_machine_learning.domain.entities.market.calibration.context_types import (
            CalibrationMethod,
            CalibrationParams,
        )

        return {
            ContextKey(self._key, 0, "ALL"): CalibrationParams(
                method=CalibrationMethod.PLATT,
                params=(1.0, 0.0),
                n_train=200,
                train_brier=0.2,
                train_ece=0.01,
            )
        }


# ─── Swap conservador (no-negociable 4) ─────────────────────────────────────


class TestConservativeSwap:
    def test_version_anterior_intacta_si_el_candidato_pierde(self):
        rng = random.Random(7)
        ctx = ContextKey("momentum", 900, "ALL")
        good_pairs = [(ctx, 0.90, rng.random() < 0.50) for _ in range(600)]

        calibrator = AdaptiveCalibrator(method=CalibrationMethod.PLATT)
        calibrator.set_version("v1")
        assert try_refit(calibrator, good_pairs) is True

        wrapper = wrap_predictor(OverconfidentPredictor(), calibrator)
        window = _window()
        before = wrapper.predict(
            window, horizon_seconds=900, observation_interval=60, lookback=20
        ).probability_up

        # Candidato imposible de mejorar: prob constante 0.5 con acierto
        # exactamente 50% en todos los segmentos ⇒ mejora de Brier 0.0
        # ⇒ REJECTED determinista en cada nivel evaluado.
        flat_pairs = [
            (ctx, 0.5, i % 2 == 0) for i in range(600)
        ]
        assert try_refit(calibrator, flat_pairs) is False

        rejected = [
            c for c in calibrator.get_comparisons().values()
            if c.verdict.value == "rejected"
        ]
        assert rejected, "el candidato debió ser rechazado"

        after = wrap_predictor(OverconfidentPredictor(), calibrator).predict(
            _window(), horizon_seconds=900, observation_interval=60, lookback=20
        ).probability_up
        # La versión activa sigue siendo v1: misma salida exacta.
        assert after == before

    def test_version_fluye_a_la_evidencia(self):
        rng = random.Random(3)
        ctx = ContextKey("momentum", 900, "ALL")
        pairs = [(ctx, 0.90, rng.random() < 0.5) for _ in range(600)]
        calibrator = AdaptiveCalibrator(method=CalibrationMethod.PLATT)
        calibrator.set_version("v4")
        assert try_refit(calibrator, pairs) is True
        wrapper = wrap_predictor(OverconfidentPredictor(), calibrator)
        wrapper.predict(
            _window(), horizon_seconds=900, observation_interval=60, lookback=20
        )
        evidence = wrapper.evidence_log[-1]
        assert evidence.calibrator_version == "v4"
        assert evidence.prob_raw == 0.90
        assert evidence.prob_calibrated != evidence.prob_raw
        assert evidence.prediction_id == "BTC-USD-momentum-1800-900"


# ─── No leakage (no-negociable 2) ───────────────────────────────────────────


class TestNoLeakage:
    def test_split_temporal_ordenado_y_disjunto(self):
        from iot_machine_learning.domain.entities.market.calibration.split import (
            train_val_test_split,
        )

        ctx = ContextKey("s", 900, "ALL")
        data = [(ctx, float(i) / 100, True) for i in range(100)]
        train, val, test = train_val_test_split(data)

        assert len(train) == 60 and len(val) == 20 and len(test) == 20
        assert train[0][1] < train[-1][1] < val[0][1] < val[-1][1] < test[0][1] < test[-1][1]
        ids = [id(d) for d in train + val + test]
        assert len(ids) == len(set(ids))

    def test_fit_solo_usa_train_outcomes_futuros_quedan_fuera(self):
        """Prueba de equivalencia exacta: el calibrador activo debe coincidir
        con uno ajustado SOLO con el segmento train. Si el fit contaminara
        con la cola (otro valor de prob y sus outcomes), el mapeo aprendido
        sería distinto."""
        from iot_machine_learning.domain.entities.market.calibration.context_calibrator import (
            ContextCalibrator,
        )
        from iot_machine_learning.domain.entities.market.calibration.verdicts import (
            FallbackLevel,
        )

        ctx = ContextKey("momentum", 900, "ALL")
        global_key = ContextKey("GLOBAL", 0, "ALL")
        early = [(ctx, 0.90, i % 2 == 0) for i in range(480)]
        late = [(ctx, 0.70, i % 3 == 0) for i in range(120)]  # otro régimen de probs

        calibrator = AdaptiveCalibrator(method=CalibrationMethod.PLATT)
        cals, comps = calibrator.train_and_evaluate(early + late)
        assert cals is not None

        # Referencia: fit aislado SOLO con train (primeras 480 filas).
        expected_calibrator = ContextCalibrator(method=CalibrationMethod.PLATT)
        expected_calibrator.fit(
            [(global_key, p, o) for _c, p, o in early[:360]]
        )

        for prob_probe in (0.90, 0.70):
            actual = calibrator.apply_with_fallback(
                ctx, prob_probe, cals
            ).prob_calibrated
            expected = expected_calibrator.calibrate(
                global_key, prob_probe
            ).prob_calibrated
            assert abs(actual - expected) < 1e-9, (
                f"fit contaminado con datos post-train para prob={prob_probe}"
            )


# ─── Evidencia: export/import ───────────────────────────────────────────────


class TestEvidencePersistence:
    def test_round_trip_preserva_campos(self):
        rng = random.Random(5)
        ctx = ContextKey("momentum", 900, "ALL")
        pairs = [(ctx, 0.90, rng.random() < 0.5) for _ in range(600)]
        calibrator = AdaptiveCalibrator(method=CalibrationMethod.PLATT)
        try_refit(calibrator, pairs)
        wrapper = wrap_predictor(OverconfidentPredictor(), calibrator)
        for hz in (300, 900):
            wrapper.predict(
                _window(),
                horizon_seconds=hz,
                observation_interval=60,
                lookback=20,
            )

        state = wrapper.export_state()
        restored = CalibratedPredictor(
            OverconfidentPredictor(), calibrator, name_suffix="-cal"
        )
        restored.import_state(state)
        assert len(restored.evidence_log) == 2
        original = list(wrapper.evidence_log)
        clone = list(restored.evidence_log)
        assert [e.to_dict() for e in original] == [e.to_dict() for e in clone]

    def test_evidence_log_acota_memoria(self):
        wrapper = wrap_predictor(
            OverconfidentPredictor(), AdaptiveCalibrator(), evidence_maxlen=3
        )
        for _ in range(5):
            wrapper.predict(
                _window(), horizon_seconds=900, observation_interval=60, lookback=20
            )
        assert len(wrapper.evidence_log) == 3
