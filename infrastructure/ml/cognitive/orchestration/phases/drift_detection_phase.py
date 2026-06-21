"""Drift Detection Phase — online concept drift detection and response.

Detects concept drift in time series using Page-Hinkley or ADWIN algorithms.
When drift is confirmed, resets BayesianWeightTracker for the affected regime
to force re-learning of engine weights.

ISO 13374 compliance: Emits DRIFT_MAGNITUDE condition indicator.
ISO 27001 compliance: Logs regime reset actions via AuditPort.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Optional

from ...drift import PageHinkleyDetector, PageHinkleyConfig, ADWINDetector

try:
    from ...observability import DriftDetectionEngine
except (ImportError, ModuleNotFoundError):
    DriftDetectionEngine = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from . import PipelineContext

logger = logging.getLogger(__name__)


class DriftDetectionPhase:
    """Phase 4: Concept drift detection and adaptive response.
    
    Monitors signal stability and noise ratio to detect concept drift.
    When drift is confirmed:
    1. Resets BayesianWeightTracker weights for the affected regime
    2. Emits ISO 13374 condition indicator
    3. Logs ISO 27001 audit trail
    4. Propagates drift metadata to downstream phases
    
    Dependency injection:
    - Detector (Page-Hinkley or ADWIN) configured via flags
    - BayesianWeightTracker injected from orchestrator
    - AuditPort injected from orchestrator
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
    ) -> None:
        """Initialize drift detection phase.

        Args:
            enable_drift_detection: Master switch for drift detection.
            drift_delta: Page-Hinkley delta parameter.
            drift_lambda: Page-Hinkley lambda threshold.
            drift_alpha: Page-Hinkley alpha forgetting factor.
            enable_adwin: Use ADWIN instead of Page-Hinkley.
            adwin_delta: ADWIN confidence parameter.
            adwin_max_window: ADWIN maximum window size.
            cooldown_seconds: Minimum seconds between drift resets per series.
            error_drift_detector: Optional ErrorDriftDetector monitoring
                prediction errors. When provided, drift is declared if EITHER
                signal-based or error-based detector triggers (conservative).
            error_drift_weight: Weight for combining signal and error drift
                scores when both are available (0=signal only, 1=error only).
        """
        self._enabled = enable_drift_detection
        self._enable_adwin = enable_adwin
        self._cooldown = cooldown_seconds
        self._last_reset: dict[str, float] = {}
        self._error_drift_detector = error_drift_detector
        self._error_drift_weight = max(0.0, min(1.0, error_drift_weight))

        # Initialize detector based on config
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
        
        # Initialize DriftDetectionEngine (Phase 3C)
        self._drift_detection_engine = drift_detection_engine
        if DriftDetectionEngine is not None and self._drift_detection_engine is None:
            self._drift_detection_engine = DriftDetectionEngine()
    
    @property
    def name(self) -> str:
        return "drift_detection"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute drift detection phase.
        
        Args:
            ctx: Pipeline context with signal profile from PerceivePhase.
        
        Returns:
            Updated context with drift metadata.
        """
        start_time = time.perf_counter()
        
        # Log phase start
        logger.info(
            "drift_detection_phase_start",
            extra={
                "phase": self.name,
                "series_id": ctx.series_id,
                "regime": ctx.regime,
                "event": "phase_start",
            },
        )
        
        # Early exit if disabled
        if not self._enabled:
            return ctx.with_field(
                drift_detected=False,
                drift_magnitude=0.0,
            )
        
        # Early exit if no profile available
        if not hasattr(ctx, 'profile') or ctx.profile is None:
            logger.warning(
                "drift_detection_no_profile",
                extra={
                    "phase": self.name,
                    "series_id": ctx.series_id,
                    "event": "WARNING",
                    "reason": "no_signal_profile_from_perceive_phase",
                    "action_taken": "skip_drift_detection",
                },
            )
            return ctx.with_field(
                drift_detected=False,
                drift_magnitude=0.0,
            )
        
        # Compute drift score from signal profile (existing behavior)
        # Values pre-sanitized by SanitizePhase[0] — see pipeline_executor.py
        signal_drift_score = self._compute_drift_score(ctx)

        # Update signal-based detector
        signal_drift_detected = self._detector.update(signal_drift_score)

        # Combine with error-based drift detection if available
        error_drift_detected = False
        error_drift_score = 0.0
        if self._error_drift_detector is not None:
            error_drift_detected = self._error_drift_detector.is_drift_detected()
            error_drift_score = self._error_drift_detector.get_drift_score()

        # Combined drift: conservative OR — drift if EITHER detector triggers.
        # This ensures we catch both signal-regime changes AND model degradation.
        drift_detected = signal_drift_detected or error_drift_detected

        # Combine scores: weighted blend of signal and error drift scores
        if self._error_drift_detector is not None:
            drift_score = (
                (1.0 - self._error_drift_weight) * signal_drift_score +
                self._error_drift_weight * error_drift_score
            )
        else:
            drift_score = signal_drift_score

        # Drift severity for conditional response in BayesianWeightTracker
        # Take the max of both severity signals
        drift_severity = max(signal_drift_score, error_drift_score)

        # Check cooldown
        now = time.time()
        series_key = f"{ctx.series_id}:{ctx.regime}"
        last_reset_time = self._last_reset.get(series_key, 0.0)

        if drift_detected and (now - last_reset_time) < self._cooldown:
            logger.info(
                "drift_detection_cooldown",
                extra={
                    "phase": self.name,
                    "series_id": ctx.series_id,
                    "event": "drift_detected_but_cooldown_active",
                    "cooldown_remaining_seconds": round(self._cooldown - (now - last_reset_time), 1),
                },
            )
            drift_detected = False

        # Reset weights if drift confirmed
        if drift_detected:
            self._handle_drift_confirmed(ctx, drift_score, now, series_key, drift_severity)
            self._detector.reset()
            if self._error_drift_detector is not None:
                self._error_drift_detector.reset()

        # Record drift metrics (Phase 3C)
        if ctx.metrics_collector is not None:
            try:
                ctx.metrics_collector.record_drift(
                    drift_detected=drift_detected,
                    drift_score=drift_score,
                )
            except Exception as e:
                logger.debug(f"metrics_collection_failed: {e}")
        
        # Use DriftDetectionEngine for multi-dimensional drift detection (Phase 3C)
        multi_dimensional_drift_result = None
        if self._drift_detection_engine is not None:
            try:
                # Prepare current distributions for drift detection
                current_regime_distribution = {ctx.regime: 1.0} if ctx.regime else {}
                current_feature_means = {
                    "mean": getattr(ctx.profile, "mean", 0.0),
                    "std": getattr(ctx.profile, "std", 0.0),
                    "slope": getattr(ctx.profile, "slope", 0.0),
                }
                current_anomaly_frequency = getattr(ctx, "consecutive_anomalies", 0) / 100.0
                current_embedding_mean = drift_score  # Use drift score as proxy
                
                multi_dimensional_drift_result = self._drift_detection_engine.detect_drift(
                    current_regime_distribution=current_regime_distribution,
                    current_feature_means=current_feature_means,
                    current_anomaly_frequency=current_anomaly_frequency,
                    current_embedding_mean=current_embedding_mean,
                    sensor_id=int(ctx.series_id) if ctx.series_id.isdigit() else None,
                    regime=ctx.regime,
                )
                
                # Combine with existing drift detection
                if multi_dimensional_drift_result.drift_detected:
                    drift_detected = True
                    logger.info(
                        "multi_dimensional_drift_detected",
                        extra={
                            "series_id": ctx.series_id,
                            "drift_type": multi_dimensional_drift_result.drift_type,
                            "drift_magnitude": multi_dimensional_drift_result.drift_magnitude,
                        },
                    )
            except Exception as e:
                logger.debug(f"multi_dimensional_drift_detection_failed: {e}")

        # ISO 13374: Emit condition indicator
        condition_indicator = {
            "name": "DRIFT_MAGNITUDE",
            "value": round(drift_score, 4),
            "unit": "dimensionless",
            "threshold": self._detector._config.lambda_ if hasattr(self._detector, '_config') else 0.0,
        }

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Log phase completion
        logger.info(
            "drift_detection_phase_complete",
            extra={
                "phase": self.name,
                "series_id": ctx.series_id,
                "event": "phase_complete",
                "result": {
                    "drift_detected": drift_detected,
                    "drift_magnitude": round(drift_score, 4),
                    "signal_drift": signal_drift_detected,
                    "error_drift": error_drift_detected,
                    "detector_used": self._detector_name,
                },
                "duration_ms": round(duration_ms, 2),
            },
        )

        return ctx.with_field(
            drift_detected=drift_detected,
            drift_magnitude=drift_score,
            condition_indicator=condition_indicator,
        )
    
    def _compute_drift_score(self, ctx: PipelineContext) -> float:
        """Compute drift score from signal profile.
        
        Combines noise ratio and inverse stability as drift proxy.
        Higher values indicate more drift.
        """
        noise_ratio = getattr(ctx.profile, 'noise_ratio', 0.0)
        stability = getattr(ctx.profile, 'stability', 1.0)
        
        # Drift score: noise + instability
        # Clamp stability to avoid division issues
        stability = max(0.01, min(1.0, stability))
        drift_score = noise_ratio + (1.0 - stability)
        
        return drift_score
    
    def _handle_drift_confirmed(
        self,
        ctx: PipelineContext,
        drift_score: float,
        now: float,
        series_key: str,
        drift_severity: float = 0.0,
    ) -> None:
        """Handle confirmed drift: reset weights and log audit trail."""
        orchestrator = ctx.orchestrator

        # ISO 27001: Capture previous state for audit
        previous_state = None
        if hasattr(orchestrator, '_plasticity') and orchestrator._plasticity:
            plasticity = orchestrator._plasticity
            if plasticity.has_history(ctx.regime):
                previous_state = {
                    "regime": ctx.regime,
                    "had_weights": True,
                }

        # Reset BayesianWeightTracker for this regime
        # Pass drift_severity for conditional response (mild vs severe)
        if hasattr(orchestrator, '_plasticity') and orchestrator._plasticity:
            try:
                orchestrator._plasticity.reset_regime(
                    regime=ctx.regime,
                    series_id=ctx.series_id,
                    drift_severity=drift_severity if drift_severity > 0 else None,
                )
                logger.warning(
                    "drift_weights_reset",
                    extra={
                        "phase": self.name,
                        "series_id": ctx.series_id,
                        "event": "WARNING",
                        "reason": "concept_drift_confirmed",
                        "action_taken": f"reset_bayesian_weights_for_regime_{ctx.regime}",
                        "drift_magnitude": round(drift_score, 4),
                    },
                )
            except Exception as e:
                logger.error(
                    "drift_reset_failed",
                    extra={
                        "phase": self.name,
                        "series_id": ctx.series_id,
                        "event": "PHASE_ERROR",
                        "error": str(e),
                        "action_taken": "continue_without_reset",
                    },
                )
        else:
            logger.error(
                "drift_no_plasticity_tracker",
                extra={
                    "phase": self.name,
                    "series_id": ctx.series_id,
                    "event": "WARNING",
                    "reason": "bayesian_weight_tracker_not_available",
                    "action_taken": "drift_detected_but_no_reset_possible",
                },
            )
        
        # ISO 27001: Log audit trail via AuditPort
        if hasattr(orchestrator, '_audit') and orchestrator._audit:
            try:
                # Use series-agnostic audit method
                orchestrator._audit.log_series_config_change(
                    series_id=ctx.series_id,
                    config_key="bayesian_weights",
                    old_value=str(previous_state) if previous_state else "none",
                    new_value="reset",
                    reason=f"concept_drift_detected_magnitude_{drift_score:.4f}",
                )
            except Exception as e:
                logger.debug(f"audit_log_failed: {e}")
        
        # Update cooldown
        self._last_reset[series_key] = now
