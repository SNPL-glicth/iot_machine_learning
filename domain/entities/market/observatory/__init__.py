"""ZENIN — Prediction Observatory (FASE 10): la memoria observable.

Prediction → Outcome → Evaluation → Reward → Dashboard | History.

ZENIN aprende → observa → se equivoca → registra → demuestra. Este
paquete es el lado de LECTURA de esa memoria: agregados puros
(``observatory.py``) y el dashboard ASCII (``render.py``). Nada aquí
escribe en el store ni toca el cerebro de ZENIN.
"""

from __future__ import annotations

from .observatory import (
    EVALUATED_STATUS,
    PENDING_STATUSES,
    BandStat,
    ContextLearning,
    DimensionStat,
    LearningPoint,
    ObservationRow,
    ObservatorySummary,
    calibration_curve,
    dimension_stats,
    evidence_requirement,
    is_degraded,
    learning_curve,
    observatory_summary,
    recency_bands,
)
from .render import render_observatory

__all__ = [
    "EVALUATED_STATUS",
    "PENDING_STATUSES",
    "BandStat",
    "ContextLearning",
    "DimensionStat",
    "LearningPoint",
    "ObservationRow",
    "ObservatorySummary",
    "calibration_curve",
    "dimension_stats",
    "evidence_requirement",
    "is_degraded",
    "learning_curve",
    "observatory_summary",
    "recency_bands",
    "render_observatory",
]
