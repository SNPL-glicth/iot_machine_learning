"""TaylorExpert — Adapter for TaylorPredictionEngine.

Patrón Adapter: No modifica TaylorEngine, solo lo envuelve.
Convierte PredictionEngine → ExpertPort para integración MoE.

SRP: Delegar al engine existente, adaptar interfaces.
"""

from __future__ import annotations

from typing import List, Optional

from iot_machine_learning.domain.ports.expert_port import ExpertPort, ExpertOutput
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.infrastructure.ml.interfaces import PredictionEngine

from ..registry.expert_capability import ExpertCapability


class TaylorExpert(ExpertPort):
    """Expert adapter for Taylor series prediction engine.
    
    Wraps TaylorPredictionEngine without modification, adapting:
    - PredictionEngine.predict(values) → ExpertPort.predict(window)
    - PredictionResult → ExpertOutput
    
    Capabilities:
    - regimes: ["volatile", "trending"] — handles non-linear dynamics
    - computational_cost: 2.5 — highest (derivative computation)
    - min_points: 5 — minimum for derivative estimation
    - specialties: ["non_linear", "derivatives", "acceleration"]
    
    Example:
        >>> from infrastructure.ml.engines.taylor import TaylorPredictionEngine
        >>> engine = TaylorPredictionEngine()
        >>> expert = TaylorExpert(engine)
        >>> 
        >>> window = SensorWindow(sensor_id=42, readings=[...])
        >>> output = expert.predict(window)
        >>> print(output.prediction, output.confidence)
    """
    
    def __init__(
        self,
        engine: PredictionEngine,
    ):
        """Initialize adapter wrapping TaylorPredictionEngine.
        
        Args:
            engine: TaylorPredictionEngine instance.
        """
        self._engine = engine
        self._capabilities = ExpertCapability(
            regimes=("volatile", "trending"),
            domains=("iot", "finance", "healthcare"),
            min_points=5,
            max_points=200,
            specialties=("non_linear", "derivatives", "acceleration"),
            computational_cost=2.5,
        )
    
    @property
    def name(self) -> str:
        """Expert identifier."""
        return "taylor"
    
    @property
    def capabilities(self) -> ExpertCapability:
        """Declared capabilities for registry matching."""
        return self._capabilities
    
    def predict(self, window: SensorWindow) -> ExpertOutput:
        """Generate prediction using wrapped TaylorEngine.
        
        Args:
            window: SensorWindow with readings.
            
        Returns:
            ExpertOutput with prediction and metadata.
        """
        # Extract values from window
        values = [r.value for r in window.readings]
        timestamps = [r.timestamp for r in window.readings] if window.readings else None
        
        # Delegate to engine
        engine_result = self._engine.predict(values, timestamps)
        
        # Adapt PredictionResult → ExpertOutput
        return ExpertOutput(
            prediction=engine_result.predicted_value,
            confidence=engine_result.confidence,
            trend=engine_result.trend,
            metadata={
                "engine_name": self._engine.name,
                "method": "taylor_series",
                **engine_result.metadata,
            },
        )
    
    def can_handle(self, window: SensorWindow) -> bool:
        """Check if window meets minimum requirements.
        
        Args:
            window: Window to evaluate.
            
        Returns:
            True if n_points >= min_points (5 for derivatives).
        """
        n_points = len(window.readings)
        return self._engine.can_handle(n_points)
    
    def estimate_latency_ms(self, n_points: int) -> float:
        """Estimate latency for given window size.
        
        Taylor is O(n) with higher constant (derivative computation).
        
        Args:
            n_points: Number of points in window.
            
        Returns:
            Estimated milliseconds.
        """
        # Taylor: ~2.0ms + 0.05ms per point (derivative computation)
        return 2.0 + (n_points * 0.05)
