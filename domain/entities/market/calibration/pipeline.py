"""FASE 10.5 — Integración de calibración en el pipeline de emisión.

Cierra el hueco entre *medir* la descalibración (dashboard 7.5, ECE/Brier)
y *corregirla* en vivo: envuelve cualquier ``Predictor`` y transforma
``probability_up`` con el ``AdaptiveCalibrator`` antes de emitir.

Contrato de honestidad:
- El wrapper es passthrough idéntico mientras no haya calibrador aceptado
  (train/val/test no alcanzado o rechazo por empeorar métricas).
- La verdad para re-entrenar son las predicciones RESUELTAS del predictor
  crudo (shadow sin calibrar). Re-calibrar sobre salidas ya calibradas
  duplicaría la corrección; ``collect_training_pairs`` existe para datos
  crudos y así se debe alimentar.
- Solo se recalibra ``probability_up``: el retorno esperado y el intervalo
  son estimaciones de magnitud, no probabilidades.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

from ..prediction.types import Regime
from ..replay.baselines import PredictionSignal, Predictor
from ..replay.classifier import classify_regime
from ..replay.feature_window import FeatureWindow
from .adaptive_calibrator import AdaptiveCalibrator
from .context_types import ContextKey

__all__ = [
    "CalibrationEvidence",
    "CalibratedPredictor",
    "wrap_predictor",
    "collect_training_pairs",
    "try_refit",
]

UNCALIBRATED = "UNCALIBRATED"


@dataclass(frozen=True)
class CalibrationEvidence:
    """Registro por predicción de qué creyó el modelo y qué corrigió el
    calibrador (FASE 10.5, no-negociable 1: raw y calibrated coexisten).

    ``prediction_id`` reconstruye el ID que emitirá el emitter para la
    misma señal (``{symbol}-{strategy}-{obs_ts}-{horizon}``), de modo que
    la evidencia puede cruzarse con la predicción persistida sin tocar el
    esquema MySQL existente.
    """

    prediction_id: str
    symbol: str
    horizon_seconds: int
    regime: str
    prob_raw: float
    prob_calibrated: float
    fallback_level: str  # valor de FallbackLevel o UNCALIBRATED
    calibrator_version: Optional[str]
    observation_timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _regime_label(regime: Regime | None) -> str:
    return regime.value if regime is not None else "ALL"


class _ResolvedPrediction(Protocol):
    """Subconjunto del contrato de ``Prediction`` que el dataset necesita."""

    strategy: str | None
    horizon_seconds: int
    regime: Regime | None
    probability_up: float
    evaluation: object | None  # Evaluation con direction_correct


def collect_training_pairs(
    predictions: Iterable[_ResolvedPrediction],
) -> List[Tuple[ContextKey, float, bool]]:
    """Construye datos de entrenamiento desde predicciones crudas resueltas.

    Solo participan predicciones con evaluación completa (direction_correct).
    La probabilidad usada es la EMITIDA: este builder está pensado para el
    histórico del predictor crudo (shadow), no para salidas ya calibradas.
    """
    pairs: List[Tuple[ContextKey, float, bool]] = []
    for pred in predictions:
        evaluation = getattr(pred, "evaluation", None)
        if evaluation is None or not hasattr(evaluation, "direction_correct"):
            continue
        direction_correct = bool(evaluation.direction_correct)
        strategy = pred.strategy or "baseline"
        context = ContextKey(
            strategy=strategy,
            horizon_seconds=pred.horizon_seconds,
            regime=_regime_label(pred.regime),
        )
        pairs.append((context, float(pred.probability_up), direction_correct))
    return pairs


def try_refit(
    calibrator: AdaptiveCalibrator,
    data: Sequence[Tuple[ContextKey, float, bool]],
) -> bool:
    """Re-entrena el calibrador; conserva el actual si el refit es rechazado.

    Devuelve True solo cuando train/val/test produjo calibradores aceptados.
    El mecanismo de rechazo (brier/economic tolerance sobre val+test
    congelados) vive en ``AdaptiveCalibrator.train_and_evaluate`` y aquí se
    respeta tal cual: None ⇒ sin cambio de estado.
    """
    calibrators, _comparisons = calibrator.train_and_evaluate(list(data))
    return calibrators is not None


class CalibratedPredictor:
    """Predictor que emite ``probability_up`` calibrada por contexto.

    Implementa el Protocol ``Predictor``, de modo que envolver
    ``cfg.predictor`` no requiere cambios en engine ni emitter:

        engine = ReplayEngine(replace(cfg, predictor=wrap_predictor(raw, cal)))

    Mientras el calibrador no tenga un nivel de fallback disponible para el
    contexto, emite la probabilidad cruda intacta (arranque honesto).
    """

    def __init__(
        self,
        inner: Predictor,
        calibrator: AdaptiveCalibrator,
        name_suffix: str = "",
        evidence_maxlen: int = 10_000,
    ) -> None:
        self._inner = inner
        self._calibrator = calibrator
        self._name_suffix = name_suffix
        self.evidence_log: Deque[CalibrationEvidence] = deque(maxlen=evidence_maxlen)
        self.last_fallback_level: str | None = None

    @property
    def name(self) -> str:
        # Sin sufijo: los IDs generados por el emitter mantienen la serie
        # del experto (upsert idempotente); con sufijo convive como experto
        # paralelo para shadow A/B.
        return f"{self._inner.name}{self._name_suffix}"

    @property
    def inner_name(self) -> str:
        """Nombre del predictor crudo (clave de contexto de calibración)."""
        return self._inner.name

    def predict(
        self,
        window: FeatureWindow,
        *,
        horizon_seconds: int,
        observation_interval: int,
        lookback: int,
    ) -> PredictionSignal:
        signal = self._inner.predict(
            window,
            horizon_seconds=horizon_seconds,
            observation_interval=observation_interval,
            lookback=lookback,
        )
        regime = classify_regime(window)
        context = ContextKey(
            strategy=self.inner_name,
            horizon_seconds=horizon_seconds,
            regime=_regime_label(regime),
        )
        version = self._calibrator.get_version()
        result = self._calibrator.apply_with_fallback(
            context, signal.probability_up, calibrator_version=version
        )
        if result.is_available:
            probability = min(0.95, max(0.05, float(result.prob_calibrated)))
            fallback_label = result.fallback_level.value
        else:
            # Sin evidencia suficiente: identidad marcada, nunca confianza
            # inventada (no-negociable 3).
            probability = signal.probability_up
            fallback_label = UNCALIBRATED
        self.last_fallback_level = fallback_label

        observation = window.last_closed()
        observation_ts = observation.timestamp if observation is not None else 0.0
        self.evidence_log.append(
            CalibrationEvidence(
                prediction_id=(
                    f"{window.symbol}-{self.name}-"
                    f"{int(observation_ts)}-{horizon_seconds}"
                ),
                symbol=window.symbol,
                horizon_seconds=horizon_seconds,
                regime=context.regime,
                prob_raw=signal.probability_up,
                prob_calibrated=probability,
                fallback_level=fallback_label,
                calibrator_version=version,
                observation_timestamp=observation_ts,
            )
        )
        return dataclasses.replace(signal, probability_up=probability)

    def export_state(self) -> Dict[str, Any]:
        """Evidencia serializable para persistirla junto al snapshot."""
        return {"evidence": [e.to_dict() for e in self.evidence_log]}

    def import_state(self, state: Dict[str, Any]) -> None:
        """Restaura la cola de evidencia (más nueva al final)."""
        self.evidence_log.clear()
        for item in state.get("evidence", []):
            self.evidence_log.append(CalibrationEvidence(**item))


def wrap_predictor(
    predictor: Predictor,
    calibrator: AdaptiveCalibrator,
    *,
    name_suffix: str = "",
    evidence_maxlen: int = 10_000,
) -> CalibratedPredictor:
    """Punto único de integración 10.5 para runners live."""
    return CalibratedPredictor(
        predictor, calibrator, name_suffix=name_suffix, evidence_maxlen=evidence_maxlen
    )


# ─── Artefacto de calibrador versionable ────────────────────────────────────
#
# El ciclo MVP es: correr UNCALIBRADO → acumular pares (prob_raw, outcome)
# en MySQL → refit offline (train/val/test con rechazo) → exportar el
# artefacto aceptado → el runner live lo carga. El artefacto es JSON puro:
# sin MySQL, sin pickle, auditable a ojo.


def _calibration_params_to_dict(params) -> Dict[str, Any]:
    return {
        "method": params.method.value,
        "params": list(params.params),
        "n_train": params.n_train,
        "train_brier": params.train_brier,
        "train_ece": params.train_ece,
    }


def export_calibrator_state(calibrator: AdaptiveCalibrator) -> Dict[str, Any]:
    """Serializa el calibrador activo (solo niveles ACCEPTED ya almacenados)."""
    from .context_types import CalibrationMethod

    levels: Dict[str, Any] = {}
    for level, context_calibrator in calibrator._calibrators.items():  # noqa: SLF001
        levels[level.value] = {
            str(context): _calibration_params_to_dict(params)
            for context, params in context_calibrator._params.items()  # noqa: SLF001
        }
    return {
        "artifact_version": 1,
        "calibrator_version": calibrator.get_version(),
        "method": calibrator.method.value,
        "levels": levels,
    }


def import_calibrator_state(
    state: Dict[str, Any],
) -> AdaptiveCalibrator:
    """Reconstruye un ``AdaptiveCalibrator`` desde ``export_calibrator_state``.

    Levanta ValueError si el artefacto está truncado o viene de otra versión
    de esquema: mejor no arrancar que arrancar calibrando a medias.
    """
    from .context_calibrator import ContextCalibrator
    from .context_types import CalibrationMethod, CalibrationParams
    from .verdicts import FallbackLevel

    if state.get("artifact_version") != 1:
        raise ValueError(
            f"artifact_version incompatible: {state.get('artifact_version')!r}"
        )
    method = CalibrationMethod(state.get("method", "platt"))
    calibrator = AdaptiveCalibrator(method=method)
    rebuilt: Dict[FallbackLevel, ContextCalibrator] = {}
    for level_name, contexts in state.get("levels", {}).items():
        level = FallbackLevel(level_name)
        context_calibrator = ContextCalibrator(method=method)
        for context_str, params_dict in contexts.items():
            strategy, horizon, regime = context_str.split("·")
            key = ContextKey(
                strategy=strategy,
                horizon_seconds=int(horizon.rstrip("s")),
                regime=regime,
            )
            params_dict = dict(params_dict)
            params_dict["method"] = CalibrationMethod(params_dict["method"])
            params_dict["params"] = tuple(params_dict["params"])
            context_calibrator._params[key] = CalibrationParams(**params_dict)  # noqa: SLF001
        rebuilt[level] = context_calibrator
    calibrator._calibrators = rebuilt  # noqa: SLF001
    version = state.get("calibrator_version")
    if version:
        calibrator.set_version(version)
    return calibrator
