"""Statistical engine adapter for Rosa Roja ExpertJury."""

from __future__ import annotations

from .base_adapter import BaseExpertAdapter
from infrastructure.ml.interfaces import PredictionEngine


class StatisticalExpertAdapter(BaseExpertAdapter):
    """Adapter for Statistical EMA/Holt engine."""
    
    def __init__(self, engine: PredictionEngine):
        super().__init__(
            engine=engine,
            name="statistical_ema_holt",
            is_critical=False,
            threshold=0.55,
            weight=0.8,
        )