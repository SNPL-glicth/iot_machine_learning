from __future__ import annotations

import warnings
from collections import defaultdict, deque
from typing import Deque, Dict, List, Optional

from ...interfaces import PredictionEngine, PredictionResult
from ..fusion import WeightedFusion, WeightMediator
from ..inhibition import InhibitionConfig, InhibitionGate
from ..plasticity import PlasticityTracker, build_advanced_plasticity, null_advanced_plasticity
from ..analysis.types import EnginePerception, MetaDiagnostic, PipelineTimer
from ..perception.record_actual_handler import record_actual_dispatch
from .pipeline_executor import execute_pipeline

_MAX_ERROR_HISTORY: int = 50


class MetaCognitiveOrchestrator(PredictionEngine):
    """Orchestrates multiple engines with cognitive reasoning.

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
    ) -> None:
        if not engines:
            raise ValueError("At least one engine required")
        self._engines = engines
        self._analyzer = SignalAnalyzer()
        self._inhibition = InhibitionGate(inhibition_config)
        self._fusion = WeightedFusion()
        self._plasticity = PlasticityTracker() if enable_plasticity else None
        if initial_weights:
            self._base_weights = dict(initial_weights)
        else:
            n = len(engines)
            self._base_weights = {e.name: 1.0 / n for e in engines}
        self._recent_errors: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=_MAX_ERROR_HISTORY)
        )
        self._last_diagnostic: Optional[MetaDiagnostic] = None
        self._last_explanation = None
        self._last_regime: Optional[str] = None
        self._last_perceptions: List[EnginePerception] = []
        self._budget_ms = budget_ms
        self._storage = storage_adapter
        self._correlation_port = correlation_port
        self._adaptive_epsilon = 0.01
        self._last_timer: Optional[PipelineTimer] = None
        
        from ..weight_adjustment_service import WeightAdjustmentService
        self._weight_service = WeightAdjustmentService(
            base_weights=self._base_weights,
            storage_adapter=storage_adapter,
            plasticity_tracker=self._plasticity,
            epsilon=self._adaptive_epsilon,
        )
        
        self._weight_mediator = WeightMediator()
        
        self._enable_advanced_plasticity = enable_advanced_plasticity
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
    ) -> PredictionResult:
        return execute_pipeline(self, values, timestamps, series_id)

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
