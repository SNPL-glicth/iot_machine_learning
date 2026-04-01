from __future__ import annotations

import warnings
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from ...interfaces import PredictionEngine, PredictionResult
from ..fusion import WeightedFusion
from ..inhibition import InhibitionConfig, InhibitionGate
from ..plasticity import PlasticityTracker, build_advanced_plasticity, null_advanced_plasticity
from ..analysis.types import EnginePerception, MetaDiagnostic, PipelineTimer
from ..analysis.signal_analyzer import SignalAnalyzer
from ..perception.record_actual_handler import record_actual_dispatch
from .pipeline_executor import execute_pipeline
from .weight_resolution_service import WeightResolutionService
from .iterative_controller import CognitiveLoopController, IterationConfig

_MAX_ERROR_HISTORY: int = 50


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
        
        # State tracking
        self._recent_errors: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=_MAX_ERROR_HISTORY)
        )
        self._last_diagnostic: Optional[MetaDiagnostic] = None
        self._last_explanation = None
        self._last_regime: Optional[str] = None
        self._last_perceptions: List[EnginePerception] = []
        self._last_timer: Optional[PipelineTimer] = None
        
        # Plasticity (learning)
        self._plasticity = PlasticityTracker() if enable_plasticity else None
        self._enable_advanced_plasticity = enable_advanced_plasticity
        
        # Build advanced plasticity components
        if enable_advanced_plasticity:
            (
                self._plasticity_coordinator,
                self._adaptive_lr,
                self._asymmetric_penalty,
                self._contextual_tracker,
                self._health_monitor,
            ) = build_advanced_plasticity(storage_adapter)
        else:
            (
                self._plasticity_coordinator,
                self._adaptive_lr,
                self._asymmetric_penalty,
                self._contextual_tracker,
                self._health_monitor,
            ) = null_advanced_plasticity()
        
        self._last_plasticity_context = None
        
        # Weight resolution (consolidated service)
        base_weights = initial_weights or {
            e.name: 1.0 / len(engines) for e in engines
        }
        self._weight_resolver = WeightResolutionService(
            base_weights=base_weights,
            plasticity_tracker=self._plasticity,
            storage_adapter=storage_adapter,
            epsilon=0.01,
        )
        
        # Storage and correlation (external ports)
        self._storage = storage_adapter
        self._correlation_port = correlation_port
        
        # Iterative cognitive loop (Phase 2)
        self._enable_iterative = enable_iterative
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
        # Capture flags once at pipeline start for consistency
        if flags_snapshot is None:
            from iot_machine_learning.ml_service.config.feature_flags import get_feature_flags
            flags_snapshot = get_feature_flags()
        
        # Use iterative cognitive loop if enabled
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
        record_actual_dispatch(
            actual_value=actual_value,
            last_regime=self._last_regime,
            last_perceptions=self._last_perceptions,
            last_plasticity_context=self._last_plasticity_context,
            enable_advanced_plasticity=self._enable_advanced_plasticity,
            plasticity_coordinator=self._plasticity_coordinator,
            plasticity_tracker=self._plasticity,
            recent_errors=self._recent_errors,
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
        return self._last_diagnostic

    @property
    def last_explanation(self):
        return self._last_explanation

    @property
    def last_pipeline_timing(self) -> Optional[PipelineTimer]:
        return self._last_timer

    # ------------------------------------------------------------------
    # Backward compatibility properties (Phase 3 refactor)
    # ------------------------------------------------------------------
    
    @property
    def _weight_service(self):
        """Backward compatibility: returns WeightResolutionService as weight service.
        
        Phase 3 refactor: WeightAdjustmentService and WeightMediator consolidated
        into WeightResolutionService. This property maintains compatibility with
        code referencing orchestrator._weight_service.
        """
        return self._weight_resolver
    
    @property
    def _weight_mediator(self):
        """Backward compatibility: returns simple pass-through mediator.
        
        Phase 3 refactor: Weight mediation logic moved into WeightResolutionService.
        This property returns a lightweight mediator that delegates to the resolver.
        """
        if not hasattr(self, '_ _compat_mediator'):
            from ..fusion import WeightMediator
            self._compat_mediator = WeightMediator()
        return self._compat_mediator
    
    @property
    def _base_weights(self) -> Dict[str, float]:
        """Backward compatibility: returns base weights from resolver."""
        if hasattr(self, '_weight_resolver'):
            return self._weight_resolver._base_weights
        return {}
