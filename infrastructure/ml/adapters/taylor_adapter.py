"""Taylor engine adapter for Rosa Roja ExpertJury."""

from __future__ import annotations

from .base_adapter import BaseExpertAdapter
from infrastructure.ml.interfaces import PredictionEngine


class TaylorExpertAdapter(BaseExpertAdapter):
    """Adapter for Taylor Finite Differences engine."""
    
    def __init__(
        self,
        engine: PredictionEngine,
        is_critical: bool = False,
        threshold: float = 0.65,
        weight: float = 1.2,
    ):
        super().__init__(
            engine=engine,
            name="taylor_finite_differences",
            is_critical=is_critical,
            threshold=threshold,
            weight=weight,
        )