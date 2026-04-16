"""BaselineExpert — Adapter for BaselineMovingAverageEngine.

Patrón Adapter: No modifica BaselineEngine, solo lo envuelve.
Convierte PredictionEngine → ExpertPort para integración MoE.

SRP: Delegar al engine existente, adaptar interfaces.
"""

from __future__ import annotations

from typing import List, Optional

from iot_machine_learning.domain.ports.expert_port import ExpertPort, ExpertOutput
from iot_machine_learning.domain.entities.sensor_reading import SensorWindow, SensorReading
from iot_machine_learning.infrastructure.ml.engines.baseline.engine import (
    predict_moving_average,
    BaselineConfig,
)
from iot_machine_learning.infrastructure.ml.interfaces import PredictionEngine

from ..registry.expert_capability import ExpertCapability


class BaselineExpert(ExpertPort):
    """Expert adapter for baseline moving average engine.
    
    Wraps BaselineEngine without modification, adapting:
    - PredictionEngine.predict(values) → ExpertPort.predict(window)
    - PredictionResult → ExpertOutput
    
    Capabilities:
    - regimes: ["stable"] — ideal for stable signals
    - computational_cost: 1.0 — baseline reference
    - min_points: 3 — minimum for meaningful average
    
    Example:
        >>> from infrastructure.ml.engines.baseline.engine import BaselineMovingAverageEngine
        >>> engine = BaselineMovingAverageEngine()
        >>> expert = BaselineExpert(engine)
        >>> 
        >>> window = SensorWindow(sensor_id=42, readings=[...])
        >>> output = expert.predict(window)
        >>> print(output.prediction, output.confidence)
    """
    
    def __init__(
        self,
        engine: PredictionEngine,
        config: Optional[BaselineConfig] = None,
    ):
        """Initialize adapter wrapping BaselineEngine.
        
        Args:
            engine: BaselineMovingAverageEngine instance.
            config: Optional config override. If None, uses engine defaults.
        """
        self._engine = engine
        self._config = config or BaselineConfig(window=20)
        self._capabilities = ExpertCapability(
            regimes=("stable",),
            domains=("iot", "finance", "healthcare"),
            min_points=3,
            max_points=1000,
            specialties=(),
            computational_cost=1.0,
        )
    
    @property
    def name(self) -> str:
        """Expert identifier."""
        return "baseline"
    
    @property
    def capabilities(self) -> ExpertCapability:
        """Declared capabilities for registry matching."""
        return self._capabilities
    
    def predict(self, window: SensorWindow) -> ExpertOutput:
        """Generate prediction using wrapped BaselineEngine.
        
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
                "window_size": self._config.window,
                **engine_result.metadata,
            },
        )
    
    def can_handle(self, window: SensorWindow) -> bool:
        """Check if window meets minimum requirements.
        
        Args:
            window: Window to evaluate.
            
        Returns:
            True if n_points >= min_points.
        """
        n_points = len(window.readings)
        return self._engine.can_handle(n_points)
    
    def estimate_latency_ms(self, n_points: int) -> float:
        """Estimate latency for given window size.
        
        Baseline is O(n) with very low constant factor.
        
        Args:
            n_points: Number of points in window.
            
        Returns:
            Estimated milliseconds.
        """
        # Baseline: ~0.01ms per point + 0.5ms overhead
        return 0.5 + (n_points * 0.01)
