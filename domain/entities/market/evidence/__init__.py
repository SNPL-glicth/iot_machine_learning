"""FASE 10.2 — Evidence Engine: evaluación multidimensional de evidencia.

Este módulo implementa el engine que decide si hay evidencia suficiente
para confiar en un contexto, considerando múltiples dimensiones más allá
de la simple accuracy (dirección):

- Sample size: n mínimo por contexto
- Direction accuracy: Wilson 95% lower bound ≥ umbral
- Magnitude quality: error promedio de magnitud ≤ tolerancia
- Economic edge: return neto después de costos ≥ mínimo
- Stability: no degradación temporal (recency bands)
- Calibration: calibración aceptable (ECE ≤ tolerancia)

El engine produce veredictos auditable con estados:
- INSUFFICIENT_EVIDENCE: no hay suficientes datos
- EVIDENCE_SUPPORTED: todas las dimensiones pasan
- EVIDENCE_DEGRADED: tenía evidencia pero degradó

Basado en el descubrimiento de 9.5: accuracy ≠ rentabilidad.
"""

from .engine import (
    EvidenceConfig,
    EvidenceDimension,
    EvidenceEngine,
    EvidenceStatus,
    EvidenceVerdict,
    render_evidence_report,
)

__all__ = [
    "EvidenceConfig",
    "EvidenceDimension",
    "EvidenceEngine",
    "EvidenceStatus",
    "EvidenceVerdict",
    "render_evidence_report",
]