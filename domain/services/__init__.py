"""Servicios de dominio UTSAE.

Orquestan la lógica de negocio usando ports (interfaces).
No conocen implementaciones concretas de infraestructura.

Subdirectories:
- prediction/ — predicción y calibración de confianza
- anomaly/ — detección de anomalías y supresión de alertas
- pattern/ — detección de patrones y coherencia
- cognitive/ — narrativa, memoria, contexto
- actions/ — recomendación de acciones
- severity/ — clasificación de severidad
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
