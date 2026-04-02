"""Data Drift Response Phase — Phase 6 UTSAE.

Automatically responds to concept drift detection by triggering
retraining alerts and adjusting plasticity parameters.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from . import PipelineContext

logger = logging.getLogger(__name__)


class DriftResponseAction:
    """Action triggered when drift exceeds threshold."""
    
    def __init__(
        self,
        name: str,
        threshold: float = 2.0,
        cooldown_seconds: float = 300.0,
    ) -> None:
        self.name = name
        self.threshold = threshold
        self.cooldown = cooldown_seconds
        self._last_triggered: Dict[str, float] = {}
    
    def should_trigger(self, series_id: str, drift_score: float, now: float) -> bool:
        """Check if action should trigger based on threshold and cooldown."""
        if drift_score < self.threshold:
            return False
        last = self._last_triggered.get(series_id, 0)
        if now - last < self.cooldown:
            return False
        self._last_triggered[series_id] = now
        return True


class DriftResponsePhase:
    """Phase X: Data Drift Response — automatic actions on concept drift.
    
    Monitors zenin_concept_drift_score and triggers:
    - Retraining alerts for IsolationForest
    - Plasticity alpha adjustment
    - Anomaly threshold recalibration
    """
    
    def __init__(
        self,
        drift_threshold: float = 2.0,
        alert_callback: Optional[Callable[[str, float, str], None]] = None,
        enable_alpha_adjustment: bool = True,
    ) -> None:
        self._threshold = drift_threshold
        self._alert_callback = alert_callback
        self._enable_alpha = enable_alpha_adjustment
        self._retraining_alert = DriftResponseAction(
            "retraining_alert",
            threshold=drift_threshold,
        )
        self._alpha_adjustment = DriftResponseAction(
            "alpha_adjustment",
            threshold=drift_threshold * 0.8,
        )
    
    @property
    def name(self) -> str:
        return "drift_response"
    
    def execute(self, ctx: PipelineContext) -> PipelineContext:
        """Execute drift response based on detected drift score."""
        import time
        
        # Get drift score from experimental metadata or compute from profile
        drift_score = self._get_drift_score(ctx)
        series_id = getattr(ctx, 'series_id', 'unknown')
        now = time.time()
        
        if drift_score < self._threshold:
            return ctx
        
        logger.warning(
            "concept_drift_detected",
            extra={
                "series_id": series_id,
                "drift_score": round(drift_score, 4),
                "threshold": self._threshold,
            },
        )
        
        # Trigger retraining alert
        if self._retraining_alert.should_trigger(series_id, drift_score, now):
            self._trigger_retraining_alert(series_id, drift_score)
        
        # Adjust plasticity alpha for faster adaptation
        if self._enable_alpha and self._alpha_adjustment.should_trigger(series_id, drift_score, now):
            self._adjust_plasticity_alpha(ctx, drift_score)
        
        # Store drift response in context
        drift_response = {
            "drift_detected": True,
            "drift_score": drift_score,
            "actions_triggered": [
                "retraining_alert" if self._retraining_alert.should_trigger(series_id, drift_score, now) else None,
                "alpha_adjustment" if self._enable_alpha else None,
            ],
        }
        
        return ctx.with_field(drift_response=drift_response)
    
    def _get_drift_score(self, ctx: PipelineContext) -> float:
        """Extract drift score from context or metadata."""
        # Check if drift was computed in experimental metadata
        if hasattr(ctx, 'experimental_metadata') and ctx.experimental_metadata:
            drift_data = ctx.experimental_metadata.get('drift', {})
            return drift_data.get('drift_score', 0.0)
        
        # Compute from signal profile if available
        if hasattr(ctx, 'profile') and ctx.profile:
            # Use noise ratio and stability as proxy for drift
            noise = getattr(ctx.profile, 'noise_ratio', 0)
            stability = getattr(ctx.profile, 'stability', 1.0)
            if noise > 0.3 and stability < 0.3:
                return 2.5  # High drift proxy
        
        return 0.0
    
    def _trigger_retraining_alert(self, series_id: str, drift_score: float) -> None:
        """Trigger alert for model retraining."""
        message = (
            f"Concept drift detected for series {series_id} "
            f"(score: {drift_score:.2f}). "
            f"Recommend retraining IsolationForest."
        )
        logger.warning(message)
        
        if self._alert_callback:
            try:
                self._alert_callback(series_id, drift_score, "retraining_required")
            except Exception as e:
                logger.error(f"alert_callback_failed: {e}")
    
    def _adjust_plasticity_alpha(self, ctx: PipelineContext, drift_score: float) -> None:
        """Adjust plasticity alpha for faster adaptation during drift."""
        if not hasattr(ctx, 'orchestrator') or not ctx.orchestrator:
            return
        
        # Get plasticity tracker if available
        plasticity = getattr(ctx.orchestrator, '_plasticity', None)
        if plasticity and hasattr(plasticity, '_ALPHA'):
            # Increase alpha temporarily for faster adaptation
            original_alpha = plasticity._ALPHA
            adjusted_alpha = min(original_alpha * 2.0, 0.5)  # Cap at 0.5
            plasticity._ALPHA = adjusted_alpha
            
            logger.info(
                "plasticity_alpha_adjusted",
                extra={
                    "original_alpha": original_alpha,
                    "adjusted_alpha": adjusted_alpha,
                    "reason": "concept_drift",
                },
            )
