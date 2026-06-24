"""Perceive Phase — signal analysis with seasonal decomposition, regime
hysteresis, and cross-equipment coherence.

Integrates SeasonalDecomposition BEFORE SignalAnalyzer so that:
  * residual is used for anomaly detection (noise ratio, std)
  * trend component slope is used for more accurate TRENDING classification
  * seasonal component enriches feature_context with seasonal_strength
    and dominant_period

Applies redis-backed regime hysteresis to prevent rapid oscillation,
and checks neighbour regime coherence for cross_regime_incoherence.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from . import PipelineContext

from iot_machine_learning.domain.value_objects.industrial_event import EventContext

logger = logging.getLogger(__name__)

_REGIME_KEY_PREFIX = "zenin:regime"
_REGIME_TTL_SECONDS = 7 * 86400


class PerceivePhase:
    """Phase [1]: signal perception with seasonal decomp + regime hysteresis."""

    name = "perceive"

    def __init__(
        self,
        hysteresis_n: int = 2,
        seasonality_enabled: bool = True,
        seasonal_min_points: int = 10,
    ) -> None:
        self._hysteresis_n = hysteresis_n
        self._seasonality_enabled = seasonality_enabled
        self._seasonal_min_points = seasonal_min_points
        self._hysteresis_cache: Dict[str, Tuple[str, int, Optional[str]]] = {}

    def execute(self, ctx: PipelineContext) -> PipelineContext:
        orchestrator = ctx.orchestrator
        series_id = ctx.series_id
        values = list(ctx.values)
        timestamps = ctx.timestamps

        # --- 1. Seasonal decomposition (inline) ---
        seasonal_strength = 0.0
        dominant_period = 0
        trend_component: Optional[List[float]] = None
        residual: Optional[List[float]] = None

        if self._seasonality_enabled and len(values) >= self._seasonal_min_points:
            seasonal_strength, dominant_period, trend_component, residual = (
                self._decompose(values, series_id)
            )

        # Use residual for analysis if we have good decomposition
        analysis_values = residual if residual and seasonal_strength > 0.2 else values

        # --- 2. Signal analysis (on residual when seasonal) ---
        profile = orchestrator._analyzer.analyze(
            analysis_values, timestamps, sensor_id=series_id,
        )

        regime_str = (
            profile.regime.value
            if hasattr(profile.regime, "value")
            else str(profile.regime)
        )

        # --- 3. More accurate TRENDING using trend component ---
        if trend_component and seasonal_strength > 0.2 and len(trend_component) >= 2:
            trend_slope = self._ols_slope(trend_component)
            if abs(trend_slope) > 1e-9:
                mean_ref = abs(profile.mean) if abs(profile.mean) > 1e-9 else 1.0
                slope_ratio = abs(trend_slope) / mean_ref
                if slope_ratio > 0.003:
                    regime_str = "TRENDING"

        # --- 4. Hour-of-day temporal context ---
        hour_of_day = None
        if timestamps:
            try:
                import datetime as _dt
                hour_of_day = _dt.datetime.fromtimestamp(timestamps[-1]).hour
            except Exception:
                pass

        # --- 5. Correlation enrichment ---
        neighbor_trends: Dict[str, str] = {}
        neighbors: List[Any] = []
        neighbor_values_dict: Dict[str, List[float]] = {}

        if orchestrator._correlation_port and series_id != "unknown":
            try:
                neighbors = orchestrator._correlation_port.get_correlated_series(
                    series_id, max_neighbors=3,
                )
                if neighbors:
                    neighbor_ids = [n[0] for n in neighbors]
                    neighbor_values_dict = (
                        orchestrator._correlation_port.get_recent_values_multi(
                            neighbor_ids, window=5,
                        )
                    )
                    for nid, nvals in neighbor_values_dict.items():
                        if len(nvals) >= 2:
                            slope = (nvals[-1] - nvals[0]) / max(len(nvals) - 1, 1)
                            neighbor_trends[nid] = (
                                "up" if slope > 0.1 else
                                "down" if slope < -0.1 else
                                "stable"
                            )
            except Exception as e:
                logger.debug(f"correlation_enrichment_failed: {e}")

        # --- 6. Regime hysteresis ---
        redis_client = self._get_redis(orchestrator)
        confirmed_regime, regime_confidence, regime_stability_score = (
            self._resolve_regime_with_hysteresis(
                series_id, regime_str, redis_client,
            )
        )

        # --- 7. Cross-regime coherence ---
        cross_regime_incoherence = False
        if neighbors:
            neighbor_regimes = self._get_neighbor_regimes(orchestrator, neighbors)
            cross_regime_incoherence = self._check_cross_regime_coherence(
                confirmed_regime, neighbor_regimes,
            )

        # --- 8. Plasticity context ---
        plasticity_context = None
        if (
            orchestrator._enable_advanced_plasticity
            and orchestrator._plasticity_coordinator
        ):
            plasticity_context = (
                orchestrator._plasticity_coordinator.create_signal_context(
                    profile, series_id,
                )
            )

        # --- 9. SensorProfile loading ---
        sensor_profile = None
        profile_repo = getattr(orchestrator, "_sensor_profile_repository", None)
        if profile_repo is not None and series_id != "unknown":
            try:
                sensor_profile = profile_repo.get_by_series_id(series_id)
            except Exception as e:
                logger.debug(f"sensor_profile_load_failed series={series_id}: {e}")

        # --- 10. Industrial event detection ---
        from iot_machine_learning.infrastructure.ml.moe.events.industrial_event_detector import (
            detect_industrial_event,
        )
        event_ctx = EventContext.none()
        try:
            event_ctx = detect_industrial_event(
                analysis_values,
                list(getattr(ctx, "sanitization_flags", [])),
                sensor_profile,
            )
            if event_ctx.is_active:
                logger.info(
                    "industrial_event_detected",
                    extra={
                        "series": series_id,
                        "event": event_ctx.detected_event.value,
                        "conf": round(event_ctx.event_confidence, 2),
                    },
                )
        except Exception as e:
            logger.debug(f"event_detection_failed: {e}")

        # --- 11. Reclassify regime with temporal context ---
        if hour_of_day is not None and sensor_profile is not None:
            from iot_machine_learning.domain.entities.series.structural_analysis import (
                _classify_regime as _cr,
            )
            enriched = _cr(
                getattr(profile, "noise_ratio", 0.0),
                getattr(profile, "slope", 0.0),
                getattr(profile, "std", 0.0),
                getattr(profile, "mean", 0.0),
                hour_of_day,
            )
            temp_regime = (
                enriched.value if hasattr(enriched, "value") else str(enriched)
            )
            # Apply hysteresis again for the reclassified regime
            confirmed_regime, regime_confidence, regime_stability_score = (
                self._resolve_regime_with_hysteresis(
                    series_id, temp_regime, redis_client,
                )
            )
            logger.debug(
                f"perceive_hour_of_day series={series_id} "
                f"hour={hour_of_day} regime={confirmed_regime}"
            )

        # --- 12. Build FeatureContext ---
        from iot_machine_learning.infrastructure.ml.moe.feature_context import (
            FeatureContext,
        )
        feature_ctx = FeatureContext.from_structural_analysis_with_profile(
            regime=confirmed_regime,
            mean=getattr(profile, "mean", 0.0),
            std=getattr(profile, "std", 0.0),
            slope=getattr(profile, "slope", 0.0),
            curvature=getattr(profile, "curvature", 0.0),
            noise_ratio=getattr(profile, "noise_ratio", 0.0),
            stability=getattr(profile, "stability", 0.0),
            hampel_outlier_mask=[],
            spatial_correlation_score=(len(neighbors) / 3.0) if neighbors else 0.0,
            sensor_profile=sensor_profile,
            event_context=event_ctx,
            seasonal_strength=seasonal_strength,
            dominant_period=dominant_period,
            regime_confidence=regime_confidence,
            cross_regime_incoherence=cross_regime_incoherence,
            regime_stability_score=regime_stability_score,
        )

        # --- 13. Metrics ---
        if ctx.metrics_collector is not None:
            try:
                ctx.metrics_collector.record_regime(confirmed_regime)
            except Exception as e:
                logger.debug(f"metrics_collection_failed: {e}")

        return ctx.with_field(
            profile=profile,
            regime=confirmed_regime,
            neighbor_trends=neighbor_trends,
            neighbors=neighbors,
            neighbor_values=neighbor_values_dict,
            plasticity_context=plasticity_context,
            feature_context=feature_ctx,
            regime_confidence=regime_confidence,
            cross_regime_incoherence=cross_regime_incoherence,
        )

    # ------------------------------------------------------------------
    # Seasonal decomposition helpers
    # ------------------------------------------------------------------

    def _decompose(
        self,
        values: List[float],
        series_id: str,
    ) -> Tuple[float, int, Optional[List[float]], Optional[List[float]]]:
        """Run FFT-based seasonal decomposition.

        Returns (seasonal_strength, dominant_period, trend, residual).
        """
        n = len(values)
        if n < self._seasonal_min_points:
            return 0.0, 0, None, None

        try:
            from ...seasonal import FFTSeasonalityDetector
            detector = FFTSeasonalityDetector(min_period=4, max_period=n // 2)
            result = detector.decompose(values)
            if result is None:
                return 0.0, 0, None, None
            trend, seasonal, residual = result
            if not seasonal or len(seasonal) != n:
                return 0.0, 0, None, None

            var_seasonal = self._variance(seasonal)
            var_original = self._variance(values)
            seasonal_strength = (
                min(var_seasonal / var_original, 1.0)
                if var_original > 1e-12
                else 0.0
            )

            period = self._detect_dominant_period(seasonal)
            return seasonal_strength, period, trend, residual
        except Exception as exc:
            logger.debug(f"seasonal_decomp_failed series={series_id}: {exc}")
            return 0.0, 0, None, None

    @staticmethod
    def _variance(values: List[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        mu = sum(values) / n
        return sum((v - mu) ** 2 for v in values) / n

    @staticmethod
    def _detect_dominant_period(seasonal: List[float]) -> int:
        """Find dominant period via autocorrelation of the seasonal component."""
        n = len(seasonal)
        if n < 4:
            return 0
        max_lag = min(n // 2, 100)
        best_lag = 0
        best_corr = -1.0
        mean_s = sum(seasonal) / n
        var_s = sum((x - mean_s) ** 2 for x in seasonal)
        if var_s < 1e-12:
            return 0
        for lag in range(2, max_lag + 1):
            x = seasonal[: n - lag]
            y = seasonal[lag:]
            if len(x) < 2:
                continue
            mx = sum(x) / len(x)
            my = sum(y) / len(y)
            num = sum((x[i] - mx) * (y[i] - my) for i in range(len(x)))
            den = math.sqrt(
                sum((xi - mx) ** 2 for xi in x)
                * sum((yi - my) ** 2 for yi in y)
            )
            if den > 1e-12:
                corr = num / den
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
        return best_lag

    @staticmethod
    def _ols_slope(values: List[float]) -> float:
        n = len(values)
        if n < 2:
            return 0.0
        t_mean = (n - 1) / 2.0
        y_mean = sum(values) / n
        num = 0.0
        den = 0.0
        for i, y in enumerate(values):
            diff_t = i - t_mean
            num += diff_t * (y - y_mean)
            den += diff_t * diff_t
        return num / den if abs(den) > 1e-15 else 0.0

    # ------------------------------------------------------------------
    # Regime hysteresis (Redis-backed with in-memory fallback)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_redis(orchestrator: Any):
        store = getattr(orchestrator, "_series_values_store", None)
        if store is not None:
            return getattr(store, "_redis", None)
        return None

    def _regime_key(self, series_id: str) -> str:
        return f"{_REGIME_KEY_PREFIX}:{series_id}"

    def _resolve_regime_with_hysteresis(
        self,
        series_id: str,
        new_regime: str,
        redis_client: Any,
    ) -> Tuple[str, float, float]:
        """Apply hysteresis: only confirm regime after N consecutive readings.

        Returns (confirmed_regime, regime_confidence, regime_stability_score).
        Redis (or in-memory fallback) stores: confirmed|counter|challenger
          * confirmed: the last confirmed regime
          * counter: how many consecutive readings of challenger
          * challenger: the regime being evaluated for confirmation
        """
        key = self._regime_key(series_id)
        raw: Optional[str] = None
        in_memory = False

        if redis_client is not None:
            try:
                stored = redis_client.get(key)
                if stored is not None:
                    raw = stored.decode() if isinstance(stored, bytes) else stored
            except Exception:
                pass

        if raw is None:
            entry = self._hysteresis_cache.get(series_id)
            if entry is not None:
                in_memory = True
                confirmed_reg, counter, challenger = entry
                raw = f"{confirmed_reg}|{counter}|{challenger if challenger else ''}"

        if not raw:
            confirmed = new_regime
            counter = 0
            challenger = None
        else:
            parts = raw.split("|")
            confirmed = parts[0] if parts[0] else new_regime
            try:
                counter = int(parts[1]) if len(parts) > 1 else 0
            except (ValueError, IndexError):
                counter = 0
            challenger_str = parts[2] if len(parts) > 2 else ""
            challenger = challenger_str if challenger_str else None

        if new_regime == confirmed:
            counter = 0
            challenger = None
            confidence = 0.9
            stability = 1.0
        elif challenger is not None and new_regime == challenger:
            counter += 1
            if counter >= self._hysteresis_n:
                confirmed = new_regime
                counter = 0
                challenger = None
                confidence = 0.9
                stability = 1.0
            else:
                confidence = 0.3 + 0.4 * (counter / self._hysteresis_n)
                stability = max(0.1, 1.0 - counter / self._hysteresis_n)
        else:
            counter = 1
            challenger = new_regime
            confidence = 0.2
            stability = 0.3

        state_str = f"{confirmed}|{counter}|{challenger if challenger else ''}"  # noqa: E501

        if redis_client is not None:
            try:
                redis_client.setex(key, _REGIME_TTL_SECONDS, state_str)
            except Exception:
                self._hysteresis_cache[series_id] = (confirmed, counter, challenger)
        else:
            self._hysteresis_cache[series_id] = (confirmed, counter, challenger)

        return confirmed, confidence, stability

    # ------------------------------------------------------------------
    # Cross-regime coherence
    # ------------------------------------------------------------------

    def _get_neighbor_regimes(
        self,
        orchestrator: Any,
        neighbors: List[Any],
    ) -> Dict[str, str]:
        """Fetch regimes of neighbour series."""
        regimes: Dict[str, str] = {}
        state_manager = getattr(orchestrator, "_context_state_manager", None)
        if state_manager is not None:
            for nbr in neighbors:
                nid = nbr[0] if isinstance(nbr, (list, tuple)) else str(nbr)
                try:
                    reg = state_manager.get_regime(nid)
                    if reg:
                        regimes[nid] = reg
                except Exception:
                    pass
        return regimes

    @staticmethod
    def _check_cross_regime_coherence(
        current_regime: str,
        neighbor_regimes: Dict[str, str],
    ) -> bool:
        """Return True if current regime is incoherent with neighbours.

        Incoherence examples:
          current=VOLATILE / STABLE and neighbour=STABLE / VOLATILE
          current=TRENDING and neighbour=STABLE
        """
        volatile_set = {"VOLATILE", "NOISY", "TRENDING"}
        stable_set = {"STABLE"}

        def _regime_group(r: str) -> int:
            r_up = r.upper().strip()
            if r_up in volatile_set:
                return 1
            if r_up in stable_set:
                return 2
            return 0

        current_group = _regime_group(current_regime)
        if current_group == 0:
            return False

        for nid, nreg in neighbor_regimes.items():
            ng = _regime_group(nreg)
            if ng == 0:
                continue
            if current_group != ng:
                return True
        return False
