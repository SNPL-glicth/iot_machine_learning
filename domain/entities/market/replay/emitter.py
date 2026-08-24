"""Market Replay Engine Prediction Emission."""

from __future__ import annotations

import dataclasses
import time

from .. import DataStatus, MarketObservation
from ..observations import Candle
from ..prediction.evaluation import evaluate_prediction
from ..prediction.lifecycle import PredictionStatus
from ..prediction.outcome import Outcome
from ..prediction.prediction import Prediction
from ..prediction.reward import RewardConfig, compute_reward
from ..prediction.types import InputContext, PredictionInterval, Regime
from .feature_window import FeatureWindow
from .clock import Clock
from .config import ReplayEngineConfig
from .classifier import classify_regime, signal_for
from .degraded import in_degraded_window


def emit_predictions(
    cfg: ReplayEngineConfig,
    observation: Candle,
    window: FeatureWindow,
    clock: Clock,
    pending: dict[int, list[Prediction]],
    resolved_by_id: dict[str, Prediction],
    emitted_order: list[Prediction],
    invalidated_order: list[Prediction],
    latency_samples: list[int],
    emit_count: int,
    latency_sample_every: int | None,
) -> int:
    """Emite predicciones para todos los horizontes configurados."""
    
    if latency_sample_every is not None:
        emit_count += 1
        sample = (emit_count % latency_sample_every) == 0
        t0 = time.perf_counter_ns() if sample else 0
    else:
        sample = False
        t0 = 0
    
    for horizon in cfg.horizons_seconds:
        signal = signal_for(horizon, window, cfg)
        strategy = cfg.strategy if cfg.strategy != "baseline" else (
            cfg.predictor.name if cfg.predictor is not None else "baseline"
        )
        if strategy == "baseline":
            prediction_id = f"{cfg.symbol}-{int(observation.timestamp)}-{horizon}"
        else:
            prediction_id = (
                f"{cfg.symbol}-{strategy}-{int(observation.timestamp)}-{horizon}"
            )
        pred = Prediction(
            prediction_id=prediction_id,
            observation=observation,
            horizon_seconds=horizon,
            timestamp=clock.now,
            entry_price=observation.close,
            expected_return=signal.expected_return,
            probability_up=signal.probability_up,
            confidence=signal.confidence_level,
            interval=PredictionInterval(
                lower=signal.lower,
                upper=signal.upper,
                confidence_level=signal.confidence_level,
            ),
            regime=classify_regime(window),
            strategy=strategy,
            input_context=InputContext(
                data_status=DataStatus.REPLAY,
                feature_count=window.size,
                feature_version="replay-v1",
            ),
        )
        # FASE 6 (modo shadow): contexto incompleto → INVALIDATED al
        # emitir, antes de cualquier resolución/reward (condición 4).
        if in_degraded_window(pred.timestamp, cfg):
            invalidated_order.append(pred.invalidate("provider_gap"))
            continue
        emitted_order.append(pred)
        resolve_ts = pred.timestamp + horizon
        pending.setdefault(resolve_ts, []).append(pred)
    
    if latency_sample_every is not None and sample:
        latency_samples.append(time.perf_counter_ns() - t0)
    
    return emit_count