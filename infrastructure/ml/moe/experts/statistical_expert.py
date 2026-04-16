"""StatisticalExpert — Adapter for StatisticalPredictionEngine.

Patrón Adapter: No modifica StatisticalEngine, solo lo envuelve.
Convierte PredictionEngine → ExpertPort para integración MoE.

SRP: Delegar al engine existente, adaptar interfaces.
"""

from __future__ import annotations

from typing import List, Optional

from iot_machine_learning.domain.ports.expert_port import ExpertPort, ExpertOutput
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow
from iot_machine_learning.infrastructure.ml.interfaces import PredictionEngine

from ..registry.expert_capability import ExpertCapability


class StatisticalExpert(ExpertPort):
    """Expert adapter for statistical EMA/Holt engine.
    
    Wraps StatisticalPredictionEngine without modification, adapting:
    - PredictionEngine.predict(values) → ExpertPort.predict(window)
    - PredictionResult → ExpertOutput
    
    Capabilities:
    - regimes: ["stable", "trending"] — handles trends via Holt's method
    - computational_cost: 1.5 — moderate (EMA computation)
    - min_points: 5 — minimum for trend detection
    
    Example:
        >>> from infrastructure.ml.engines.statistical import StatisticalPredictionEngine
        >>> engine = StatisticalPredictionEngine()
        >>> expert = StatisticalExpert(engine)
        >>> 
        >>> window = SensorWindow(sensor_id=42, readings=[...])
        >>> output = expert.predict(window)
        >>> print(output.prediction, output.confidence)
    """
    
    def __init__(
        self,
        engine: PredictionEngine,
    ):
        """Initialize adapter wrapping StatisticalPredictionEngine.
        
        Args:
            engine: StatisticalPredictionEngine instance.
        """
        self._engine = engine
        self._capabilities = ExpertCapability(
            regimes=("stable", "trending"),
            domains=("iot", "finance"),
            min_points=5,
            max_points=500,
            specialties=("seasonality", "trend_detection"),
            computational_cost=1.5,
        )
    
    @property
    def name(self) -> str:
        """Expert identifier."""
        return "statistical"
    
    @property
    def capabilities(self) -> ExpertCapability:
        """Declared capabilities for registry matching."""
        return self._capabilities
    
    def predict(self, window: SensorWindow) -> ExpertOutput:
        """Generate prediction using wrapped StatisticalEngine.
        
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
                "method": "ema_holt",
                **engine_result.metadata,
            },
        )
    
    def can_handle(self, window: SensorWindow) -> bool:
        """Check if window meets minimum requirements.
        
        Args:
            window: Window to evaluate.
            
        Returns:
            True if n_points >= min_points (5 for trend detection).
        """
        n_points = len(window.readings)
        return self._engine.can_handle(n_points)
    
    def estimate_latency_ms(self, n_points: int) -> float:
        """Estimate latency for given window size.
        
        Statistical is O(n) with moderate constant (EMA computation).
        
        Args:
            n_points: Number of points in window.
            
        Returns:
            Estimated milliseconds.
        """
        # Statistical: ~1.0ms + 0.02ms per point
        return 1.0 + (n_points * 0.02)
