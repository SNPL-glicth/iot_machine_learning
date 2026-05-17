from __future__ import annotations

import threading
import warnings
from typing import Any, Dict, List, Optional

from ...interfaces import PredictionEngine, PredictionResult
from ..error_store import EngineErrorStore
from ..fusion import WeightedFusion
from ..inhibition import InhibitionConfig, InhibitionGate
from ..reliability import EngineReliabilityTracker
from ..hyperparameters import HyperparameterAdaptor
from ..series_values import SeriesValuesStore
from ..bayesian_weight_tracker import BayesianWeightTracker, build_advanced_bayesian, null_advanced_bayesian
from ..analysis.types import EnginePerception, MetaDiagnostic, PipelineTimer
from ..analysis.signal_analyzer import SignalAnalyzer
from ..perception.record_actual_handler import record_actual_dispatch
from .pipeline_executor import execute_pipeline
from .pipeline_executor_factory import PipelineExecutorFactory
from .weight_resolution_service import WeightResolutionService
from .iterative_controller import CognitiveLoopController, IterationConfig
from .error_history_manager import create_error_history_manager
from .context_state_manager import ContextStateManager

from iot_machine_learning.infrastructure.ml.moe.engine_weight_initializer import (
    EquipmentTypeWeightInitializer,
    StructuralEngineFilter,
)
from iot_machine_learning.infrastructure.security.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitScope,
    RateLimitExceeded,
)
from iot_machine_learning.infrastructure.security.rate_limit_decorator import rate_limit_series


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
        # ARCH: Config BUDGET_MS=25000 (25s pipeline max) vs this guard=500ms
        # The 500ms guard fires early for single-phase budget; 25s is total cycle.
        budget_ms: float = 500.0,
        storage_adapter=None,
        enable_advanced_plasticity: bool = False,
        correlation_port=None,
        enable_iterative: bool = False,
        redis_client: Optional[Any] = None,
        hyperparameter_adaptor: Optional[HyperparameterAdaptor] = None,
        series_values_store: Optional[SeriesValuesStore] = None,
        compliance_exporter: Optional[Any] = None,
        rate_limiter: Optional[RateLimiter] = None,
        enable_rate_limiting: bool = True,
        sensor_profile_repository: Optional[Any] = None,
        weight_initializer: Optional[Any] = None,
        engine_filter: Optional[Any] = None,
    ) -> None:
        if not engines:
            raise ValueError("At least one engine required")

        # IMP-4a: single canonical error bus (Redis or in-memory).
        self._error_store = EngineErrorStore(redis_client=redis_client)
        # IMP-4b: Beta-Bernoulli reliability tracker feeds InhibitionGate
        # and replaces the three hardcoded threshold rules.
        self._reliability_tracker = EngineReliabilityTracker(
            error_store=self._error_store,
            redis_client=redis_client,
        )
        # IMP-4c: single source of truth for per-series hyperparameters
        # (Redis-only; inert when redis_client is None).
        self._hyperparameter_adaptor = hyperparameter_adaptor or HyperparameterAdaptor(
            redis_client=redis_client,
        )
        # IMP-1: rolling raw-values buffer consumed by SanitizePhase.
        # Inert when no redis_client supplied. Read by the phase from
        # ctx.orchestrator._series_values_store.
        self._series_values_store = series_values_store or SeriesValuesStore(
            redis_client=redis_client,
        )
        # IMP-3: fresh PipelineExecutor per predict() — no singleton race.
        # PIPE-2: compliance_exporter injected from app.state (None if disabled).
        self._pipeline_executor_factory = PipelineExecutorFactory(
            compliance_exporter=compliance_exporter,
        )
        
        # SEC-4: Rate limiter for DoS prevention
        self._rate_limiter = rate_limiter
        self._enable_rate_limiting = enable_rate_limiting and rate_limiter is not None
        self._correlation_port = correlation_port
        self._sensor_profile_repository = sensor_profile_repository
        self._weight_initializer = weight_initializer or EquipmentTypeWeightInitializer()
        self._engine_filter = engine_filter or StructuralEngineFilter()
        from iot_machine_learning.infrastructure.ml.moe.events.prediction_drift_detector import PredictionDriftDetector
        self._drift_detector = PredictionDriftDetector()

        # Core components
        self._engines = engines
        # IMP-4c: share the adaptor with engines that expose a ``_hyperparams``
        # slot and were built without one. Explicit adaptors set on the engine
        # take precedence and are left untouched.
        for _engine in self._engines:
            if getattr(_engine, "_hyperparams", "missing") is None:
                _engine._hyperparams = self._hyperparameter_adaptor
        self._analyzer = SignalAnalyzer()
        self._inhibition = InhibitionGate(
            inhibition_config,
            reliability_tracker=self._reliability_tracker,
        )
        self._fusion = WeightedFusion()
        self._budget_ms = budget_ms

        # R-1: Per-series state isolation via ContextStateManager
        self._state_manager = ContextStateManager(max_series=10000)
        self._error_history = create_error_history_manager(
            max_history=50,
            error_store=self._error_store,
        )
        
        # GOLD: Initialize storage adapter (was missing - critical bug fix)
        self._storage = storage_adapter
        
        # Thread-safe locks for non-series-specific state
        self._state_lock = threading.RLock()
        self._last_diagnostic: Optional[MetaDiagnostic] = None
        self._last_explanation = None
        self._last_timer: Optional[PipelineTimer] = None
        
        # Bayesian weight tracking (learning) - must be before weight_resolver
        # IMP-4a: share the single EngineErrorStore so the 99p cap is
        # sourced from the canonical error bus (no second copy).
        self._plasticity = (
            BayesianWeightTracker(error_store=self._error_store)
            if enable_plasticity else None
        )
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
        series_id: str,
        values: List[float],
        timestamps: Optional[List[float]] = None,
        **kwargs,
    ) -> PredictionResult:
        """Predict with rate limiting.
        
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        # SEC-4: Check rate limits before processing
        if self._enable_rate_limiting:
            # Check series-level rate limit
            result = self._rate_limiter.check_series_limit(
                series_id=series_id,
            )
            
            if not result.allowed:
                raise RateLimitExceeded(
                    result=result,
                    scope=RateLimitScope.SERIES,
                    identifier=series_id,
                )
        
        # Continue with normal prediction
        return self._predict_internal(series_id, values, timestamps, **kwargs)
    
    def _predict_internal(
        self,
        series_id: str,
        values: List[float],
        timestamps: Optional[List[float]] = None,
        **kwargs,
    ) -> PredictionResult:
        if 'flags_snapshot' not in kwargs:
            raise ValueError(
                "flags_snapshot is required. Pass feature flags via constructor "
                "or flags_snapshot parameter to enable testable injection."
            )

        flags_snapshot = kwargs['flags_snapshot']

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
            last_signal_context=series_state.last_signal_context,
            enable_advanced_plasticity=self._enable_advanced_plasticity,
            plasticity_coordinator=self._plasticity_coordinator,
            plasticity_tracker=self._plasticity,
            error_history=self._error_history,
            storage=self._storage,
            series_id=series_id,
            series_context=series_context,
            reliability_tracker=self._reliability_tracker,
            engines=self._engines,
            orchestrator_ref=self,
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
    # ------------------------------------------------------------------
