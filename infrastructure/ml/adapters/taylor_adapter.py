"""Taylor engine adapter for Rosa Roja ExpertJury."""

from __future__ import annotations

from .base_adapter import BaseExpertAdapter
from infrastructure.ml.interfaces import PredictionEngine


class TaylorExpertAdapter(BaseExpertAdapter):
    """Adapter for Taylor Finite Differences engine."""
    
    def __init__(self, engine: PredictionEngine):
        super().__init__(
            engine=engine,
            name="taylor_finite_differences",
            is_critical=True,
            threshold=0.65,
            weight=1.2,
        )


def create_taylor_adapter(**engine_kwargs) -> TaylorExpertAdapter:
    """Factory to create Taylor adapter with engine."""
    from infrastructure.ml.engines.taylor.engine import TaylorPredictionEngine
    engine = TaylorPredictionEngine(**engine_kwargs)
    return TaylorExpertAdapter(engine)