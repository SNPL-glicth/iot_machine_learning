"""Advanced Bayesian Coordinator.

Coordinates the 4 components of the advanced Bayesian weight tracking system.
Renamed from 'plasticity' for honest naming.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .context_builder import build_plasticity_context
from .error_persister import ErrorPersister

logger = logging.getLogger(__name__)


class AdvancedBayesianCoordinator:
    """Coordinates advanced Bayesian weight tracking components."""
    
    def __init__(
        self,
        adaptive_lr,
        asymmetric_penalty,
        contextual_tracker,
        health_monitor,
        storage_adapter=None,
    ):
        self._adaptive_lr = adaptive_lr
        self._asymmetric_penalty = asymmetric_penalty
        self._contextual_tracker = contextual_tracker
        self._health_monitor = health_monitor
        self._persister = ErrorPersister(storage_adapter)
        self._legacy_tracker = None
    
    def set_legacy_tracker(self, tracker) -> None:
        """Attach legacy PlasticityTracker for fallback."""
        self._legacy_tracker = tracker

    def get_weights(self, regime: str, engine_names: List[str], context=None) -> Dict[str, float]:
        """Return normalized weights. Contextual takes priority."""
        if self._contextual_tracker and context is not None:
            weights = self._contextual_tracker.get_contextual_weights(
                series_id="_global", engine_names=engine_names, context=context,
            )
            if weights is not None:
                return weights
        
        if self._legacy_tracker and self._legacy_tracker.has_history(regime):
            return self._legacy_tracker.get_weights(regime, engine_names)
        
        n = len(engine_names)
        return {name: 1.0 / n for name in engine_names} if n > 0 else {}

    def record_error(self, engine_name: str, error: float, regime: str, context=None) -> None:
        """Record error to contextual tracker and legacy."""
        if self._contextual_tracker and context is not None:
            self._contextual_tracker.record_error("_global", engine_name, error, context)
        if self._legacy_tracker:
            self._legacy_tracker.update(regime, engine_name, error)

    def has_history(self, regime: str) -> bool:
        """True if any plasticity data exists."""
        return self._legacy_tracker.has_history(regime) if self._legacy_tracker else False

    def create_plasticity_context(self, profile, series_id: str):
        """Create PlasticityContext from signal profile."""
        return build_plasticity_context(profile, series_id)

    def record_actual_advanced(
        self,
        actual_value: float,
        perceptions: List,
        plasticity_context,
        regime: str,
        series_id: str,
        series_context=None,
        plasticity_tracker=None,
        error_history=None,
    ) -> None:
        """Record actual value with advanced plasticity."""
        if not series_id or not plasticity_context:
            return
        
        errors_dict = {p.engine_name: abs(p.predicted_value - actual_value) for p in perceptions}
        mean_error = sum(errors_dict.values()) / len(errors_dict) if errors_dict else 0.0
        
        for p in perceptions:
            self._process_engine(
                p, actual_value, errors_dict[p.engine_name], mean_error,
                plasticity_context, regime, series_id, series_context,
                plasticity_tracker, error_history
            )
    
    def _process_engine(
        self, p, actual, error, mean_error, context, regime, series_id,
        series_ctx, tracker, err_history,
    ):
        """Process single engine prediction."""
        # Asymmetric penalty
        penalty = self._asymmetric_penalty.compute_penalty(
            p.predicted_value, actual, error, series_ctx,
        ) if self._asymmetric_penalty and series_ctx else error
        
        # Consensus penalty
        adj_error = penalty * (1.0 + 0.5 * abs(error - mean_error) / (mean_error + 1e-6))
        
        # Adaptive LR
        lr = self._adaptive_lr.compute_adaptive_lr(adj_error, context) if self._adaptive_lr else 0.15
        
        # Record to history
        if err_history:
            err_history.record_error(series_id, p.engine_name, error)
        
        # Update legacy
        if tracker:
            tracker.update(regime, p.engine_name, error, alpha=lr)
        
        # Contextual error
        if self._contextual_tracker:
            self._contextual_tracker.record_error(series_id, p.engine_name, error, context)
        
        # Health update
        if self._health_monitor:
            state = self._health_monitor.record_prediction(series_id, p.engine_name, error)
            if state.is_inhibited:
                logger.warning("engine_inhibited", extra={"series_id": series_id, "engine": p.engine_name})
            self._persister.persist_engine_health(series_id, p.engine_name, state)
        
        # Persist error
        self._persister.persist_contextual_error(
            series_id, p.engine_name, p.predicted_value, actual, error, penalty, context,
        )
