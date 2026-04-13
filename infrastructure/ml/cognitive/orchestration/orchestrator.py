from __future__ import annotations

import threading
import warnings
from typing import List, Optional

from ...interfaces import PredictionEngine, PredictionResult
from ..fusion import WeightedFusion
from ..inhibition import InhibitionConfig, InhibitionGate
from ..bayesian_weight_tracker import BayesianWeightTracker, build_advanced_bayesian, null_advanced_bayesian
from ..analysis.types import EnginePerception, MetaDiagnostic, PipelineTimer
from ..analysis.signal_analyzer import SignalAnalyzer
from ..perception.record_actual_handler import record_actual_dispatch
from .pipeline_executor import execute_pipeline
from .weight_resolution_service import WeightResolutionService
from .iterative_controller import CognitiveLoopController, IterationConfig
from .error_history_manager import create_error_history_manager
from .context_state_manager import ContextStateManager


class MetaCognitiveOrchestrator(PredictionEngine):
    """Orchestrates multiple engines with cognitive reasoning.

    Simplified Phase 3 design:
    - Weight resolution consolidated into WeightResolutionService
    - Reduced from 15+ dependencies to core pipeline components
    - Clear separation: orchestration vs weight resolution vs plasticity
    
    Phase 2: Added optional iterative cognitive loop for confidence-based refinement.

    Implements PredictionEngine — drop-in replacement for any single engine.
    """

    def __init__(
        self,
        engines: List[PredictionEngine],
        initial_weights: Optional[Dict[str, float]] = None,
        inhibition_config: Optional[InhibitionConfig] = None,
        enable_plasticity: bool = True,
        budget_ms: float = 500.0,
        storage_adapter=None,
        enable_advanced_plasticity: bool = False,
        correlation_port=None,
        enable_iterative: bool = False,
    ) -> None:
        if not engines:
            raise ValueError("At least one engine required")
        
        # Core components
        self._engines = engines
        self._analyzer = SignalAnalyzer()
        self._inhibition = InhibitionGate(inhibition_config)
        self._fusion = WeightedFusion()
        self._budget_ms = budget_ms
        
        # R-1: Per-series state isolation via ContextStateManager
        self._state_manager = ContextStateManager(max_series=10000)
        self._error_history = create_error_history_manager(max_history=50)
        
        # GOLD: Initialize storage adapter (was missing - critical bug fix)
        self._storage = storage_adapter
        
        # Thread-safe locks for non-series-specific state
        self._state_lock = threading.RLock()
        self._last_diagnostic: Optional[MetaDiagnostic] = None
        self._last_explanation = None
        self._last_timer: Optional[PipelineTimer] = None
        
        # Bayesian weight tracking (learning) - must be before weight_resolver
        self._plasticity = BayesianWeightTracker() if enable_plasticity else None
        self._enable_advanced_plasticity = enable_advanced_plasticity
        
        # GOLD: Weight resolution service (consolidated from Phase 3 refactor)
        base_weights = initial_weights or {e.name: 1.0 / len(engines) for e in engines}
        self._weight_resolver = WeightResolutionService(
            base_weights=base_weights,
            plasticity_tracker=self._plasticity,
            storage_adapter=storage_adapter,
        )
        
        # Build advanced Bayesian weight tracking components
        if enable_advanced_plasticity:
            (
                self._plasticity_coordinator,
                self._adaptive_lr,
                self._asymmetric_penalty,
                self._contextual_tracker,
                self._health_monitor,
            ) = build_advanced_bayesian(storage_adapter)
        else:
            (
                self._plasticity_coordinator,
                self._adaptive_lr,
                self._asymmetric_penalty,
                self._contextual_tracker,
                self._health_monitor,
            ) = null_advanced_bayesian()
        
        # Iterative cognitive loop
        self._enable_iterative = enable_iterative
        self._loop_controller = None
        if enable_iterative:
            self._loop_controller = CognitiveLoopController(
                pipeline_fn=execute_pipeline,
                config=IterationConfig(
                    max_iterations=3,
                    confidence_threshold=0.85,
                    time_budget_ms=5000.0,
                ),
            )
        else:
            self._loop_controller = None

    @property
    def name(self) -> str:
        return "meta_cognitive_orchestrator"

    def can_handle(self, n_points: int) -> bool:
        return any(e.can_handle(n_points) for e in self._engines)

    def predict(
        self,
        values: List[float],
        timestamps: Optional[List[float]] = None,
        series_id: str = "unknown",
        flags_snapshot=None,
    ) -> PredictionResult:
        if flags_snapshot is None:
            raise ValueError(
                "flags_snapshot is required. Pass feature flags via constructor "
                "or flags_snapshot parameter to enable testable injection."
            )
        
        # Track prediction count for this specific series
        self._state_manager.increment_prediction_count(series_id)
        
        if self._enable_iterative and self._loop_controller is not None:
            return self._loop_controller.execute(
                orchestrator=self,
                values=values,
                timestamps=timestamps,
                series_id=series_id,
                flags_snapshot=flags_snapshot,
            )
        
        # Standard single-pass pipeline (backward compatible)
        return execute_pipeline(self, values, timestamps, series_id, flags_snapshot=flags_snapshot)

    def record_actual(
        self,
        actual_value: float,
        series_id: Optional[str] = None,
        series_context=None,
    ) -> None:
        """Record actual value with series-isolated state (R-1)."""
        if series_id is None:
            raise ValueError("series_id is required for state isolation (R-1)")
        
        # R-1: Get series-specific state instead of shared
        series_state = self._state_manager.get_state(series_id)
        
        record_actual_dispatch(
            actual_value=actual_value,
            last_regime=series_state.last_regime,
            last_perceptions=series_state.last_perceptions,
            last_plasticity_context=series_state.last_plasticity_context,
            enable_advanced_plasticity=self._enable_advanced_plasticity,
            plasticity_coordinator=self._plasticity_coordinator,
            plasticity_tracker=self._plasticity,
            error_history=self._error_history,
            storage=self._storage,
            series_id=series_id,
            series_context=series_context,
        )

    @property
    def last_diagnostic(self) -> Optional[MetaDiagnostic]:
        warnings.warn(
            "last_diagnostic is deprecated. Use last_explanation instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # GOLD: Protect with state lock to prevent race conditions
        with self._state_lock:
            return self._last_diagnostic

    @property
    def last_explanation(self):
        # GOLD: Protect with state lock to prevent race conditions
        with self._state_lock:
            return self._last_explanation

    @property
    def last_pipeline_timing(self) -> Optional[PipelineTimer]:
        # GOLD: Protect with state lock to prevent race conditions  
        with self._state_lock:
            return self._last_timer

    @property
    def state_manager(self) -> ContextStateManager:
        """Access state manager for testing and monitoring."""
        return self._state_manager

    # ------------------------------------------------------------------
    # GOLD: Removed broken backward compatibility properties
    # Phase 3 refactor consolidated weight services into WeightResolutionService
    # Use _weight_resolver directly or the public orchestrator interface
    # ------------------------------------------------------------------
