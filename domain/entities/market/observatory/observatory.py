"""ZENIN — Prediction Observatory (FASE 10): la memoria observable.

ZENIN aprende → observa → se equivoca → registra → demuestra. Este
módulo es el lado de LECTURA de esa memoria: recibe las filas
persistidas del ciclo Prediction → Outcome → Evaluation → Reward y
produce los agregados del dashboard, sin tocar el cerebro ni la
adaptación:

- resumen por estado (total / evaluadas / pendientes / invalidadas /
  archivadas / stale);
- accuracy + Wilson 95% por dimensión (horizonte, estrategia, régimen);
- curva de calibración reutilizando ``bucket_calibration`` (FASE 7.5):
  el bucket 0.7 que acierta 49% es FAIL → el modelo miente ahí;
- learning curve por contexto (experto × horizonte × régimen): accuracy
  acumulada en n observaciones ordenadas por tiempo;
- requerimiento de evidencia: el n mínimo donde el Wilson 95% de la
  accuracy acumulada cruza un umbral → "necesito ≈ N observaciones
  antes de considerar a este experto";
- bandas de recencia: ¿la accuracy del pasado reciente es peor que la
  del antiguo? (degradación).

Regla de la fase: ZENIN NO toca plata todavía; primero demuestra con
memoria. Nada aquí escribe: solo agrega filas del store.
"""

from __future__ import annotations

from .types import (
    ObservationRow,
    ObservatorySummary,
    DimensionStat,
    LearningPoint,
    ContextLearning,
    BandStat,
)
from .summary import observatory_summary
from .dimension import dimension_stats
from .calibration import calibration_curve
from .learning import learning_curve, evidence_requirement
from .recency import recency_bands, is_degraded
from .helpers import EVALUATED_STATUS, PENDING_STATUSES

__all__ = [
    "EVALUATED_STATUS",
    "PENDING_STATUSES",
    "ObservationRow",
    "ObservatorySummary",
    "DimensionStat",
    "LearningPoint",
    "ContextLearning",
    "BandStat",
    "observatory_summary",
    "dimension_stats",
    "calibration_curve",
    "learning_curve",
    "evidence_requirement",
    "recency_bands",
    "is_degraded",
]