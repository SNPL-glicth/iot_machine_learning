"""Drift Detection Phase — online concept drift detection with Redis
persistence, EWMA gradual drift, and drift cause classification.

Detects abrupt drift (Page-Hinkley) and gradual drift (EWMA sustained
threshold).  Classifies drift cause as sensor_degradation,
operational_change, seasonal_shift, or unknown.  Propagates enriched
drift_event into PredictionResult metadata and caps max_action when
appropriate.

Redis keys:
  zenin:drift:reset:{series_id}:{regime}   — last reset timestamp (TTL 24h)
  zenin:drift:ph_state:{series_id}          — PageHinkley serialised state (TTL 24h)
  zenin:metrics:drift_count:{series_id}     — monotonic counter (INCR)
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from ...drift import PageHinkleyDetector, PageHinkleyConfig, ADWINDetector

try:
    from ...observability import DriftDetectionEngine
except (ImportError, ModuleNotFoundError):
    DriftDetectionEngine = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from . import PipelineContext

logger = logging.getLogger(__name__)

_DRIFT_RESET_KEY_PREFIX = "zenin:drift:reset"
_DRIFT_PH_STATE_PREFIX = "zenin:drift:ph_state"
_DRIFT_COUNTER_PREFIX = "zenin:metrics:drift_count"
_DRIFT_TTL_SECONDS = 86400

_EWMA_WINDOW = 20
_EWMA_ALPHA = 2.0 / (_EWMA_WINDOW + 1)
_EWMA_THRESHOLD = 0.6
_EWMA_CONSECUTIVE_REQUIRED = 5


def _get_redis(orchestrator: Any):
    store = getattr(orchestrator, "_series_values_store", None)
    if store is not None:
        return getattr(store, "_redis", None)
    return None


class DriftDetectionPhase:
    """Phase 4: concept drift detection with Redis-backed persistence,
    gradual drift via EWMA, and drift cause classification.
    """

    def __init__(
        self,
        enable_drift_detection: bool = True,
        drift_delta: float = 0.005,
        drift_lambda: float = 50.0,
        drift_alpha: float = 0.9999,
        enable_adwin: bool = False,
        adwin_delta: float = 0.002,
        adwin_max_window: int = 1000,
        cooldown_seconds: float = 300.0,
        error_drift_detector: Optional[Any] = None,
        error_drift_weight: float = 0.5,
        drift_detection_engine: Optional[Any] = None,
        gradual_ewma_window: int = _EWMA_WINDOW,
        gradual_threshold: float = _EWMA_THRESHOLD,
        gradual_consecutive_required: int = _EWMA_CONSECUTIVE_REQUIRED,
    ) -> None:
        self._enabled = enable_drift_detection
        self._enable_adwin = enable_adwin
        self._cooldown = cooldown_seconds
        self._error_drift_detector = error_drift_detector
        self._error_drift_weight = max(0.0, min(1.0, error_drift_weight))

        # Gradual drift (EWMA) config
        self._ewma_window = gradual_ewma_window
        self._ewma_alpha = 2.0 / (gradual_ewma_window + 1)
        self._gradual_threshold = gradual_threshold
        self._gradual_consecutive = gradual_consecutive_required

        # Per-series EWMA state (in-memory; short window, repopulates fast)
        self._ewma_value: Dict[str, float] = {}
        self._ewma_consecutive: Dict[str, int] = {}

        # In-memory fallback for last_reset (Redis is primary)
        self._last_reset_fallback: Dict[str, float] = {}

        # In-memory fallback for persisted PH state
        self._ph_state_fallback: Dict[str, str] = {}

        # Initialise detector
        if enable_adwin:
            self._detector = ADWINDetector(
                delta=adwin_delta,
                max_window_size=adwin_max_window,
            )
            self._detector_name = "adwin"
        else:
            config = PageHinkleyConfig(
                delta=drift_delta,
                lambda_=drift_lambda,
                alpha=drift_alpha,
            )
            self._detector = PageHinkleyDetector(config)
            self._detector_name = "page_hinkley"

        # DriftDetectionEngine (Phase 3C)
        self._drift_detection_engine = drift_detection_engine
        if DriftDetectionEngine is not None and self._drift_detection_engine is None:
            self._drift_detection_engine = DriftDetectionEngine()

    @property
    def name(self) -> str:
        return "drift_detection"

    # ── Redis helpers ─────────────────────────────────────────────

    @staticmethod
    def _reset_key(series_id: str, regime: str) -> str:
        return f"{_DRIFT_RESET_KEY_PREFIX}:{series_id}:{regime}"

    @staticmethod
    def _ph_state_key(series_id: str) -> str:
        return f"{_DRIFT_PH_STATE_PREFIX}:{series_id}"

    @staticmethod
    def _counter_key(series_id: str) -> str:
        return f"{_DRIFT_COUNTER_PREFIX}:{series_id}"

    def _redis_load_reset(self, redis_client: Any, series_key: str) -> float:
        if redis_client is not None:
            try:
                raw = redis_client.get(series_key)
                if raw is not None:
                    return float(raw.decode() if isinstance(raw, bytes) else raw)
            except Exception:
                pass
        return self._last_reset_fallback.get(series_key, 0.0)

    def _redis_save_reset(self, redis_client: Any, series_key: str, ts: float) -> None:
        if redis_client is not None:
            try:
                redis_client.setex(series_key, _DRIFT_TTL_SECONDS, str(ts))
            except Exception:
                self._last_reset_fallback[series_key] = ts
        else:
            self._last_reset_fallback[series_key] = ts

    def _redis_load_ph_state(self, redis_client: Any, series_id: str) -> Optional[str]:
        if redis_client is not None:
            try:
                raw = redis_client.get(self._ph_state_key(series_id))
                if raw is not None:
                    return raw.decode() if isinstance(raw, bytes) else raw
            except Exception:
                pass
        return self._ph_state_fallback.get(series_id)

    def _redis_save_ph_state(
        self, redis_client: Any, series_id: str, state: str
    ) -> None:
        if redis_client is not None:
            try:
                redis_client.setex(
                    self._ph_state_key(series_id), _DRIFT_TTL_SECONDS, state
                )
            except Exception:
                self._ph_state_fallback[series_id] = state
        else:
            self._ph_state_fallback[series_id] = state

    def _redis_increment_counter(self, redis_client: Any, series_id: str) -> None:
        if redis_client is not None:
            try:
                redis_client.incr(self._counter_key(series_id))
            except Exception:
                pass

    # ── PH state serialisation ──────────────────────────────────

    @staticmethod
    def _serialise_ph_state(sum_: float, mean: float, n: int) -> str:
        return f"{sum_}|{mean}|{n}"

    @staticmethod
    def _deserialise_ph_state(
        raw: str,
    ) -> Tuple[float, float, int]:
        if not raw:
            return 0.0, 0.0, 0
        parts = raw.split("|")
        sum_ = float(parts[0]) if len(parts) > 0 and parts[0] else 0.0
        mean = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
        n = int(parts[2]) if len(parts) > 2 and parts[2] else 0
        return sum_, mean, n

    def _restore_ph_state(self, series_id: str) -> None:
        """Restore PageHinkley detector state from Redis (or in-memory fallback)."""
        redis_client = None
        # Try to get redis from detector's ephemeral access — we accept None
        raw = self._redis_load_ph_state(redis_client, series_id)
        if raw:
            sum_, mean, n = self._deserialise_ph_state(raw)
            self._detector._sum = sum_
            self._detector._mean = mean
            self._detector._n = n

    def _save_ph_state(self, series_id: str) -> None:
        redis_client = None
        state = self._serialise_ph_state(
            self._detector._sum, self._detector._mean, self._detector._n
        )
        self._redis_save_ph_state(redis_client, series_id, state)

    # ── EWMA gradual drift ──────────────────────────────────────

    def _update_ewma(
        self, series_id: str, drift_score: float
    ) -> Tuple[float, int]:
        prev = self._ewma_value.get(series_id, drift_score)
        ewma = self._ewma_alpha * drift_score + (1.0 - self._ewma_alpha) * prev
        self._ewma_value[series_id] = ewma

        if ewma > self._gradual_threshold:
            count = self._ewma_consecutive.get(series_id, 0) + 1
        else:
            count = 0
        self._ewma_consecutive[series_id] = count
        return ewma, count

    # ── Cause classification ─────────────────────────────────────

    def _classify_drift_cause(
        self,
        ctx: PipelineContext,
        drift_score: float,
    ) -> str:
        features = ctx.feature_context

        # Check if neighbours are stable
        neighbours_stable = self._neighbours_stable(ctx)

        # Check seasonal shift
        seasonal_shift = (
            features is not None
            and features.seasonal_strength > 0.3
            and features.dominant_period > 0
        )

        if neighbours_stable and not seasonal_shift:
            return "sensor_degradation"

        if seasonal_shift:
            return "seasonal_shift"

        # If neighbours also show instability, likely operational
        if not neighbours_stable:
            return "operational_change"

        return "unknown"

    @staticmethod
    def _neighbours_stable(ctx: PipelineContext) -> bool:
        regimes: Dict[str, str] = {}
        state_mgr = getattr(ctx.orchestrator, "_context_state_manager", None)
        neighbors = getattr(ctx, "neighbors", None)
        if state_mgr is not None and neighbors:
            for nbr in neighbors:
                nid = nbr[0] if isinstance(nbr, (list, tuple)) else str(nbr)
                try:
                    reg = state_mgr.get_regime(nid)
                    if reg:
                        regimes[nid] = reg
                except Exception:
                    pass
        if not regimes:
            return True
        stable_count = sum(
            1 for r in regimes.values() if r.upper().strip() == "STABLE"
        )
        return stable_count >= len(regimes) * 0.5

    # ── Drift score ──────────────────────────────────────────────

    def _compute_drift_score(self, ctx: PipelineContext) -> float:
        noise_ratio = getattr(ctx.profile, "noise_ratio", 0.0)
        stability = getattr(ctx.profile, "stability", 1.0)
        stability = max(0.01, min(1.0, stability))
        return noise_ratio + (1.0 - stability)

    # ── Cooldown ─────────────────────────────────────────────────

    def _check_cooldown(
        self,
        redis_client: Any,
        series_id: str,
        regime: str,
    ) -> Tuple[bool, float]:
        series_key = self._reset_key(series_id, regime)
        last_ts = self._redis_load_reset(redis_client, series_key)
        now = time.time()
        if last_ts > 0 and (now - last_ts) < self._cooldown:
            return True, self._cooldown - (now - last_ts)
        return False, 0.0

    # ── Drift confirmation handler ───────────────────────────────

    def _handle_drift_confirmed(
        self,
        ctx: PipelineContext,
        drift_score: float,
        drift_severity: float,
        drift_type: str,
        drift_cause: str,
    ) -> None:
        orchestrator = ctx.orchestrator
        redis_client = _get_redis(orchestrator)

        now = time.time()

        # Reset BayesianWeightTracker
        if hasattr(orchestrator, "_plasticity") and orchestrator._plasticity:
            try:
                orchestrator._plasticity.reset_regime(
                    regime=ctx.regime,
                    series_id=ctx.series_id,
                    drift_severity=drift_severity if drift_severity > 0 else None,
                )
                logger.warning(
                    "drift_weights_reset",
                    extra={
                        "series_id": ctx.series_id,
                        "event": "WARNING",
                        "reason": "concept_drift_confirmed",
                        "drift_type": drift_type,
                        "drift_cause": drift_cause,
                        "drift_magnitude": round(drift_score, 4),
                    },
                )
            except Exception as e:
                logger.error(
                    "drift_reset_failed",
                    extra={
                        "series_id": ctx.series_id,
                        "event": "PHASE_ERROR",
                        "error": str(e),
                    },
                )
        else:
            logger.error(
                "drift_no_plasticity_tracker",
                extra={
                    "series_id": ctx.series_id,
                    "event": "WARNING",
                    "reason": "bayesian_weight_tracker_not_available",
                },
            )

        # ISO 27001 audit trail
        if hasattr(orchestrator, "_audit") and orchestrator._audit:
            try:
                orchestrator._audit.log_series_config_change(
                    series_id=ctx.series_id,
                    config_key="bayesian_weights",
                    old_value="none",
                    new_value="reset",
                    reason=(
                        f"drift_detected_type={drift_type}_cause={drift_cause}"
                        f"_magnitude={drift_score:.4f}"
                    ),
                )
            except Exception as e:
                logger.debug(f"audit_log_failed: {e}")

        # Persist last_reset
        series_key = self._reset_key(ctx.series_id, ctx.regime)
        self._redis_save_reset(redis_client, series_key, now)

        # Persist PH state (so detector resumes after restart)
        state = self._serialise_ph_state(
            self._detector._sum, self._detector._mean, self._detector._n
        )
        self._redis_save_ph_state(redis_client, ctx.series_id, state)

        # Increment dashboard counter
        self._redis_increment_counter(redis_client, ctx.series_id)

        # Build drift_event for metadata
        drift_event = {
            "type": drift_type,
            "magnitude": round(drift_score, 4),
            "cause": drift_cause,
            "affected_regime": ctx.regime,
            "timestamp": now,
            "detector": self._detector_name,
        }

        # Store drift_event in metadata
        ctx.metadata["drift_event"] = drift_event

        # Cap max_action based on drift cause
        if drift_cause == "sensor_degradation":
            current_action = getattr(ctx, "max_action", "PREDICT")
            if current_action in ("ESCALATE", "PREDICT"):
                ctx.with_field(max_action="INVESTIGATE")

        logger.warning(
            "drift_confirmed",
            extra={
                "series_id": ctx.series_id,
                "drift_type": drift_type,
                "drift_cause": drift_cause,
                "drift_magnitude": round(drift_score, 4),
            },
        )

    # ── Main execute ─────────────────────────────────────────────

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        start_time = time.perf_counter()

        logger.info(
            "drift_detection_phase_start",
            extra={
                "phase": self.name,
                "series_id": ctx.series_id,
                "regime": ctx.regime,
            },
        )

        if not self._enabled:
            return ctx.with_field(
                drift_detected=False,
                drift_magnitude=0.0,
                drift_type=None,
                drift_cause=None,
            )

        if not hasattr(ctx, "profile") or ctx.profile is None:
            logger.warning(
                "drift_detection_no_profile",
                extra={
                    "phase": self.name,
                    "series_id": ctx.series_id,
                    "reason": "no_signal_profile",
                },
            )
            return ctx.with_field(
                drift_detected=False,
                drift_magnitude=0.0,
                drift_type=None,
                drift_cause=None,
            )

        redis_client = _get_ctx_redis(ctx)

        # Restore PH state from Redis on first call for this series
        # (we check if detector has zero observations to detect "fresh" state)
        if self._detector._n == 0:
            self._restore_ph_state_from_redis(redis_client, ctx.series_id)

        # Compute drift score
        signal_drift_score = self._compute_drift_score(ctx)

        # --- Abrupt drift (Page-Hinkley / ADWIN) ---
        signal_drift_detected = self._detector.update(signal_drift_score)

        # --- Gradual drift (EWMA) ---
        ewma_val, ewma_consecutive = self._update_ewma(
            ctx.series_id, signal_drift_score
        )
        gradual_drift_detected = ewma_consecutive >= self._gradual_consecutive

        # --- Error drift ---
        error_drift_detected = False
        error_drift_score = 0.0
        if self._error_drift_detector is not None:
            error_drift_detected = self._error_drift_detector.is_drift_detected()
            error_drift_score = self._error_drift_detector.get_drift_score()

        # Combined drift score
        if self._error_drift_detector is not None:
            drift_score = (
                (1.0 - self._error_drift_weight) * signal_drift_score
                + self._error_drift_weight * error_drift_score
            )
        else:
            drift_score = signal_drift_score

        drift_severity = max(signal_drift_score, error_drift_score)

        # Determine drift type
        drift_type: Optional[str] = None
        if signal_drift_detected:
            drift_type = "abrupt"
        elif gradual_drift_detected:
            drift_type = "gradual"

        drift_detected = drift_type is not None or error_drift_detected

        # Cooldown check
        in_cooldown, _ = self._check_cooldown(
            redis_client, ctx.series_id, ctx.regime or "UNKNOWN"
        )
        if drift_detected and in_cooldown:
            logger.info(
                "drift_detection_cooldown",
                extra={
                    "series_id": ctx.series_id,
                    "event": "drift_detected_but_cooldown_active",
                },
            )
            drift_detected = False
            drift_type = None

        # Handle confirmed drift
        drift_cause: Optional[str] = None
        if drift_detected:
            drift_cause = self._classify_drift_cause(ctx, drift_score)
            self._handle_drift_confirmed(
                ctx, drift_score, drift_severity, drift_type or "unknown", drift_cause
            )
            # Reset detectors after handling
            self._detector.reset()
            if self._error_drift_detector is not None:
                self._error_drift_detector.reset()

        # Persist PH state after each update (for restart recovery)
        self._save_ph_state_to_redis(redis_client, ctx.series_id)

        # Record metrics
        if ctx.metrics_collector is not None:
            try:
                ctx.metrics_collector.record_drift(
                    drift_detected=drift_detected,
                    drift_score=drift_score,
                )
            except Exception as e:
                logger.debug(f"metrics_collection_failed: {e}")

        # Multi-dimensional drift (Phase 3C)
        if self._drift_detection_engine is not None:
            try:
                current_regime_distribution = (
                    {ctx.regime: 1.0} if ctx.regime else {}
                )
                current_feature_means = {
                    "mean": getattr(ctx.profile, "mean", 0.0),
                    "std": getattr(ctx.profile, "std", 0.0),
                    "slope": getattr(ctx.profile, "slope", 0.0),
                }
                current_anomaly_frequency = (
                    getattr(ctx, "consecutive_anomalies", 0) / 100.0
                )
                current_embedding_mean = drift_score

                multi_result = self._drift_detection_engine.detect_drift(
                    current_regime_distribution=current_regime_distribution,
                    current_feature_means=current_feature_means,
                    current_anomaly_frequency=current_anomaly_frequency,
                    current_embedding_mean=current_embedding_mean,
                    sensor_id=(
                        int(ctx.series_id) if ctx.series_id.isdigit() else None
                    ),
                    regime=ctx.regime,
                )
                if multi_result.drift_detected and not drift_detected:
                    drift_detected = True
                    drift_type = drift_type or "multi_dimensional"
                    logger.info(
                        "multi_dimensional_drift_detected",
                        extra={
                            "series_id": ctx.series_id,
                            "drift_type": multi_result.drift_type,
                            "drift_magnitude": multi_result.drift_magnitude,
                        },
                    )
            except Exception as e:
                logger.debug(f"multi_dimensional_drift_failed: {e}")

        # ISO 13374 condition indicator
        condition_indicator = {
            "name": "DRIFT_MAGNITUDE",
            "value": round(drift_score, 4),
            "unit": "dimensionless",
            "threshold": (
                self._detector._config.lambda_
                if hasattr(self._detector, "_config")
                else 0.0
            ),
        }

        duration_ms = (time.perf_counter() - start_time) * 1000

        logger.info(
            "drift_detection_phase_complete",
            extra={
                "phase": self.name,
                "series_id": ctx.series_id,
                "duration_ms": round(duration_ms, 2),
                "drift_detected": drift_detected,
                "drift_type": drift_type,
                "drift_cause": drift_cause,
                "drift_magnitude": round(drift_score, 4),
            },
        )

        return ctx.with_field(
            drift_detected=drift_detected,
            drift_magnitude=drift_score,
            drift_type=drift_type,
            drift_cause=drift_cause,
            condition_indicator=condition_indicator,
        )

    def _restore_ph_state_from_redis(
        self, redis_client: Any, series_id: str
    ) -> None:
        raw = self._redis_load_ph_state(redis_client, series_id)
        if raw:
            sum_, mean, n = self._deserialise_ph_state(raw)
            if n > 0:
                self._detector._sum = sum_
                self._detector._mean = mean
                self._detector._n = n
                logger.info(
                    "drift_ph_state_restored",
                    extra={
                        "series_id": series_id,
                        "n_observations": n,
                        "mean": round(mean, 4),
                    },
                )

    def _save_ph_state_to_redis(
        self, redis_client: Any, series_id: str
    ) -> None:
        state = self._serialise_ph_state(
            self._detector._sum,
            self._detector._mean,
            self._detector._n,
        )
        self._redis_save_ph_state(redis_client, series_id, state)


def _get_ctx_redis(ctx: PipelineContext) -> Any:
    return _get_redis(ctx.orchestrator)
