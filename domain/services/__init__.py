"""Servicios de dominio UTSAE.

Orquestan la lógica de negocio usando ports (interfaces).
No conocen implementaciones concretas de infraestructura.

Subdirectories:
- actions/ — recomendación de acciones
- anomaly/ — detección de anomalías y supresión de alertas
- calibration/ — calibración de evidencia y predicciones
- cognitive/ — narrativa, memoria, contexto
- manifold/ — geometría diferencial y variedades
- pattern/ — detección de patrones y coherencia
- prediction/ — predicción y toma de decisiones
- semantic_extraction/ — extracción de entidades y priorización
- severity/ — clasificación de severidad
- takens/ — reconstrucción de espacio de fases y dimensión espectral
"""
try:
    from .prediction.prediction_domain_service import PredictionDomainService
except ImportError:
    PredictionDomainService = None  # type: ignore[assignment,misc]

try:
    from .anomaly.anomaly_domain_service import AnomalyDomainService
except ImportError:
    AnomalyDomainService = None  # type: ignore[assignment,misc]

try:
    from .pattern.pattern_domain_service import PatternDomainService
except ImportError:
    PatternDomainService = None  # type: ignore[assignment,misc]

__all__ = [
    "PredictionDomainService",
    "AnomalyDomainService",
    "PatternDomainService",
]
