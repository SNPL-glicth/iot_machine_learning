"""Advanced Plasticity Coordinator.

Coordinates the 4 components of the advanced plasticity system:
- AdaptiveLearningRate: Context-aware learning rates
- AsymmetricPenaltyService: Business-impact penalties
- ContextualPlasticityTracker: Context-specific MAE tracking
- EngineHealthMonitor: Auto-inhibition

Extracted from MetaCognitiveOrchestrator to reduce complexity.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import List, Optional

logger = logging.getLogger(__name__)


class AdvancedPlasticityCoordinator:
    """Coordinates advanced plasticity components.
    
    Responsibilities:
    - Create PlasticityContext from signal profile
    - Coordinate error recording across 4 components
    - Manage adaptive learning rate calculation
    - Handle asymmetric penalty application
    - Coordinate storage persistence
    
    Thread Safety:
        All coordinated components are thread-safe with RLock.
    """
    
    def __init__(
        self,
        adaptive_lr,
        asymmetric_penalty,
        contextual_tracker,
        health_monitor,
        storage_adapter=None,
    ):
        """Initialize coordinator with plasticity components.
        
        Args:
            adaptive_lr: AdaptiveLearningRate instance
            asymmetric_penalty: AsymmetricPenaltyService instance
            contextual_tracker: ContextualPlasticityTracker instance
            health_monitor: EngineHealthMonitor instance
            storage_adapter: Optional storage for persistence
        """
        self._adaptive_lr = adaptive_lr
        self._asymmetric_penalty = asymmetric_penalty
        self._contextual_tracker = contextual_tracker
        self._health_monitor = health_monitor
        self._storage = storage_adapter
    
    def create_plasticity_context(self, profile, series_id: str):
        """Create PlasticityContext from signal profile.
        
        Args:
            profile: Signal profile from analyzer
            series_id: Series identifier
        
        Returns:
            PlasticityContext with regime, volatility, noise, time
        """
        from ....domain.entities.plasticity.plasticity_context import (
            PlasticityContext,
            RegimeType,
        )
        
        # Map regime string to RegimeType enum
        regime_str = profile.regime.value if hasattr(profile, 'regime') else 'unknown'
        regime_map = {
            "stable": RegimeType.STABLE,
            "trending": RegimeType.TRENDING,
            "volatile": RegimeType.VOLATILE,
            "noisy": RegimeType.NOISY,
        }
        regime_type = regime_map.get(regime_str.lower(), RegimeType.UNKNOWN)
        
        # Get volatility from profile
        volatility = getattr(profile, 'volatility', 0.0)
        if hasattr(profile, 'volatility_level'):
            # Map VolatilityLevel to float
            vol_map = {"low": 0.2, "medium": 0.5, "high": 0.8}
            volatility = vol_map.get(profile.volatility_level.value.lower(), 0.5)
        
        # Get noise ratio (estimate from signal quality)
        noise_ratio = 1.0 - getattr(profile, 'signal_quality', 0.5)
        
        # Create context
        now = datetime.now()
        return PlasticityContext(
            regime=regime_type,
            noise_ratio=min(max(noise_ratio, 0.0), 1.0),
            volatility=min(max(volatility, 0.0), 1.0),
            time_of_day=now.hour,
            consecutive_failures=0,  # Will be updated by health monitor
            error_magnitude=0.0,  # Will be calculated in record_actual
            is_critical_zone=False,  # Will be determined from series_context
            timestamp=now,
        )
    
    def record_actual_advanced(
        self,
        actual_value: float,
        perceptions: List,
        plasticity_context,
        regime: str,
        series_id: str,
        series_context=None,
        plasticity_tracker=None,
        recent_errors=None,
    ) -> None:
        """Record actual value with advanced plasticity system.
        
        Integrates all 4 components:
        1. Calculate base error
        2. Apply asymmetric penalty
        3. Calculate adaptive learning rate
        4. Update legacy plasticity with adaptive lr
        5. Record contextual error
        6. Update engine health
        7. Persist to database (contextual error)
        8. Persist to database (engine health)
        
        Args:
            actual_value: True observed value
            perceptions: List of EnginePerception from prediction
            plasticity_context: PlasticityContext for this prediction
            regime: Signal regime string
            series_id: Series identifier
            series_context: Optional SeriesContext for asymmetric penalty
            plasticity_tracker: Optional legacy PlasticityTracker
            recent_errors: Optional dict of recent errors
        """
        if not series_id or not plasticity_context:
            return
        
        for p in perceptions:
            # 1. Calculate base error
            error = abs(p.predicted_value - actual_value)
            
            # 2. Apply asymmetric penalty (FASE 2)
            penalty = error
            if self._asymmetric_penalty and series_context:
                penalty = self._asymmetric_penalty.compute_penalty(
                    predicted=p.predicted_value,
                    actual=actual_value,
                    base_error=error,
                    context=series_context,
                )
            
            # 3. Calculate adaptive learning rate (FASE 1)
            learning_rate = 0.15  # Default
            if self._adaptive_lr:
                learning_rate = self._adaptive_lr.compute_adaptive_lr(
                    error=penalty,
                    context=plasticity_context,
                )
            
            # 4. Update legacy plasticity with adaptive lr
            if recent_errors is not None:
                recent_errors[p.engine_name].append(error)
            
            if plasticity_tracker is not None:
                # Override alpha with adaptive learning rate
                plasticity_tracker._alpha = learning_rate
                plasticity_tracker.update(regime, p.engine_name, error)
            
            # 5. Record contextual error (FASE 3)
            if self._contextual_tracker:
                self._contextual_tracker.record_error(
                    series_id=series_id,
                    engine_name=p.engine_name,
                    error=error,
                    context=plasticity_context,
                )
            
            # 6. Update engine health (FASE 4)
            if self._health_monitor:
                state = self._health_monitor.record_prediction(
                    series_id=series_id,
                    engine_name=p.engine_name,
                    error=error,
                )
                
                # Log inhibition events
                if state.is_inhibited:
                    logger.warning(
                        "engine_auto_inhibited",
                        extra={
                            "series_id": series_id,
                            "engine_name": p.engine_name,
                            "reason": state.inhibition_reason,
                            "consecutive_failures": state.consecutive_failures,
                        },
                    )
            
            # 7. Persist to database (FASE 6) - contextual error
            if self._storage and hasattr(self._storage, 'record_contextual_error'):
                try:
                    self._storage.record_contextual_error(
                        series_id=series_id,
                        engine_name=p.engine_name,
                        predicted_value=p.predicted_value,
                        actual_value=actual_value,
                        error=error,
                        penalty=penalty,
                        regime=plasticity_context.regime.value,
                        noise_ratio=plasticity_context.noise_ratio,
                        volatility=plasticity_context.volatility,
                        time_of_day=plasticity_context.time_of_day,
                        consecutive_failures=plasticity_context.consecutive_failures,
                        is_critical_zone=plasticity_context.is_critical_zone,
                        context_key=plasticity_context.context_key,
                    )
                except Exception as e:
                    logger.error(
                        "record_contextual_error_failed",
                        extra={"series_id": series_id, "error": str(e)},
                    )
            
            # 8. Update engine health in database
            if self._storage and self._health_monitor and hasattr(self._storage, 'update_engine_health'):
                try:
                    state = self._health_monitor.get_state(series_id, p.engine_name)
                    if state:
                        self._storage.update_engine_health(
                            series_id=series_id,
                            engine_name=p.engine_name,
                            consecutive_failures=state.consecutive_failures,
                            consecutive_successes=state.consecutive_successes,
                            total_predictions=state.total_predictions,
                            total_errors=state.total_errors,
                            last_error=state.last_error,
                            failure_rate=state.failure_rate,
                            is_inhibited=state.is_inhibited,
                            inhibition_reason=state.inhibition_reason,
                            last_success_time=state.last_success_time.isoformat() if state.last_success_time else None,
                            last_failure_time=state.last_failure_time.isoformat() if state.last_failure_time else None,
                        )
                except Exception as e:
                    logger.error(
                        "update_engine_health_failed",
                        extra={"series_id": series_id, "error": str(e)},
                    )
