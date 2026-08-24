"""FASE 10.4 — Drift Detection: vigilancia continua de degradación temporal.

Este módulo implementa detección de drift para monitorear cambios
en performance temporal de ZENIN:

- DriftDetector: detector con múltiples métodos (window comparison, Page-Hinkley)
- WindowMetrics: métricas por ventana temporal (Last 100, 500, 1,000, 5,000, Lifetime)
- DriftAlert: alertas cuando se detecta degradación significativa
- Conexión con sistema Page-Hinkley existente
- Diferenciación entre ruido normal y drift real

Objetivo: detectar cuando ZENIN empieza a degradarse para tomar
acciones (reentrenar, cambiar estrategia, etc.) antes de perder dinero.
"""

from .detector import (
    DriftAlert,
    DriftConfig,
    DriftDetector,
    DriftStatus,
    WindowMetrics,
    page_hinkley_test,
    render_drift_report,
)

__all__ = [
    "DriftAlert",
    "DriftConfig",
    "DriftDetector",
    "DriftStatus",
    "WindowMetrics",
    "page_hinkley_test",
    "render_drift_report",
]