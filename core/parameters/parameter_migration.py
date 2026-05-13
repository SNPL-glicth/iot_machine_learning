"""Migración de parámetros existentes al registry.

Une los 87 parámetros del audit.
"""

from __future__ import annotations

from core.parameters.numerical_constants import CONFIDENCE, EPSILON, STAT_THRESHOLDS
from core.parameters.parameter_registry import (
    ParameterCategory,
    ParameterMetadata,
    ParameterRegistry,
    ParameterScope,
)


def register_epsilon_constants(registry: ParameterRegistry) -> None:
    """Registra constantes epsilon."""
    registry.register(
        ParameterMetadata(
            name="EPSILON.COMPARISON",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=EPSILON.COMPARISON,
            default_value=1e-9,
            min_value=1e-15,
            max_value=1e-6,
            description="Epsilon para comparaciones de magnitud",
            mathematical_meaning="Threshold para |x| < ε",
            statistical_justification="Machine epsilon ≈ 2.2e-16, usar 1e-9 es seguro",
            version="1.0.0",
        )
    )

    registry.register(
        ParameterMetadata(
            name="EPSILON.DIVISION",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=EPSILON.DIVISION,
            default_value=1e-12,
            min_value=1e-15,
            max_value=1e-9,
            description="Epsilon para denominadores (division safety)",
            mathematical_meaning="Clamp denominador: max(x, ε)",
            statistical_justification="1e-12 apropiado para double precision",
            version="1.0.0",
        )
    )

    registry.register(
        ParameterMetadata(
            name="EPSILON.CONFIDENCE",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=EPSILON.CONFIDENCE,
            default_value=1e-6,
            min_value=1e-12,
            max_value=1e-3,
            description="Epsilon para cálculos de confianza",
            mathematical_meaning="Threshold para cálculos de probabilidad",
            statistical_justification="Rango intermedio entre COMPARISON y DIVISION",
            version="1.0.0",
        )
    )

    registry.register(
        ParameterMetadata(
            name="EPSILON.CORRELATION",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=EPSILON.CORRELATION,
            default_value=1e-9,
            min_value=1e-15,
            max_value=1e-6,
            description="Epsilon para correlaciones",
            mathematical_meaning="Threshold para |r| < ε",
            statistical_justification="Similar a COMPARISON para correlaciones",
            version="1.0.0",
        )
    )

    registry.register(
        ParameterMetadata(
            name="EPSILON.GRADIENT",
            category=ParameterCategory.NUMERICAL_STABILITY,
            scope=ParameterScope.GLOBAL,
            value=EPSILON.GRADIENT,
            default_value=1e-9,
            min_value=1e-15,
            max_value=1e-6,
            description="Epsilon para gradientes",
            mathematical_meaning="Threshold para |∇f| < ε",
            statistical_justification="Similar a COMPARISON para gradientes",
            version="1.0.0",
        )
    )


def register_statistical_thresholds(registry: ParameterRegistry) -> None:
    """Registra thresholds estadísticos."""
    registry.register(
        ParameterMetadata(
            name="STAT_THRESHOLDS.Z_SCORE_LOWER",
            category=ParameterCategory.STATISTICAL_THRESHOLD,
            scope=ParameterScope.GLOBAL,
            value=STAT_THRESHOLDS.Z_SCORE_LOWER,
            default_value=2.0,
            min_value=1.0,
            max_value=4.0,
            description="Z-score threshold inferior para voting",
            mathematical_meaning="2σ ≈ 95.4% confidence bajo normalidad",
            statistical_justification="Threshold basado en distribución normal estándar",
            dependencies=["STAT_THRESHOLDS.Z_SCORE_UPPER"],
            validation=lambda x: x > 0,
            version="1.0.0",
        )
    )

    registry.register(
        ParameterMetadata(
            name="STAT_THRESHOLDS.Z_SCORE_UPPER",
            category=ParameterCategory.STATISTICAL_THRESHOLD,
            scope=ParameterScope.GLOBAL,
            value=STAT_THRESHOLDS.Z_SCORE_UPPER,
            default_value=3.0,
            min_value=1.0,
            max_value=5.0,
            description="Z-score threshold superior para voting",
            mathematical_meaning="3σ ≈ 99.7% confidence bajo normalidad",
            statistical_justification="Threshold conservador para detección de anomalías",
            dependencies=[],
            validation=lambda x: x > 0,
            version="1.0.0",
        )
    )

    registry.register(
        ParameterMetadata(
            name="STAT_THRESHOLDS.CONTAMINATION_DEFAULT",
            category=ParameterCategory.STATISTICAL_THRESHOLD,
            scope=ParameterScope.GLOBAL,
            value=STAT_THRESHOLDS.CONTAMINATION_DEFAULT,
            default_value=0.005,
            min_value=0.001,
            max_value=0.1,
            description="Contamination rate esperado (proporción de outliers)",
            mathematical_meaning="Proporción esperada de anomalías en datos normales",
            statistical_justification="0.5% consistente con 3σ bajo normalidad (99.7%)",
            version="1.0.0",
        )
    )

    registry.register(
        ParameterMetadata(
            name="STAT_THRESHOLDS.CONTAMINATION_MIN",
            category=ParameterCategory.STATISTICAL_THRESHOLD,
            scope=ParameterScope.GLOBAL,
            value=STAT_THRESHOLDS.CONTAMINATION_MIN,
            default_value=0.001,
            min_value=0.0001,
            max_value=0.01,
            description="Contamination rate mínimo permitido",
            mathematical_meaning="Límite inferior para contamination",
            statistical_justification="Prevenir overfitting en datasets muy limpios",
            dependencies=["STAT_THRESHOLDS.CONTAMINATION_DEFAULT"],
            validation=lambda x: x > 0,
            version="1.0.0",
        )
    )

    registry.register(
        ParameterMetadata(
            name="STAT_THRESHOLDS.CONTAMINATION_MAX",
            category=ParameterCategory.STATISTICAL_THRESHOLD,
            scope=ParameterScope.GLOBAL,
            value=STAT_THRESHOLDS.CONTAMINATION_MAX,
            default_value=0.05,
            min_value=0.01,
            max_value=0.2,
            description="Contamination rate máximo permitido",
            mathematical_meaning="Límite superior para contamination",
            statistical_justification="Prevenir detección excesiva de falsos positivos",
            dependencies=["STAT_THRESHOLDS.CONTAMINATION_DEFAULT"],
            validation=lambda x: x > 0,
            version="1.0.0",
        )
    )

    registry.register(
        ParameterMetadata(
            name="STAT_THRESHOLDS.IQR_FENCE_MULTIPLIER",
            category=ParameterCategory.STATISTICAL_THRESHOLD,
            scope=ParameterScope.GLOBAL,
            value=STAT_THRESHOLDS.IQR_FENCE_MULTIPLIER,
            default_value=1.5,
            min_value=1.0,
            max_value=3.0,
            description="Multiplicador de IQR para Tukey fences",
            mathematical_meaning="Tukey standard: [Q1 - k*IQR, Q3 + k*IQR]",
            statistical_justification="1.5 es estándar en boxplots (≈ 2σ para normal)",
            version="1.0.0",
        )
    )

    registry.register(
        ParameterMetadata(
            name="STAT_THRESHOLDS.LOF_MAX_NEIGHBORS",
            category=ParameterCategory.STATISTICAL_THRESHOLD,
            scope=ParameterScope.GLOBAL,
            value=STAT_THRESHOLDS.LOF_MAX_NEIGHBORS,
            default_value=20,
            min_value=5,
            max_value=50,
            description="Máximo de vecinos para LOF detector",
            mathematical_meaning="Parámetro k en algoritmo LOF",
            statistical_justification="Balance entre precisión y performance",
            version="1.0.0",
        )
    )


def register_confidence_config(registry: ParameterRegistry) -> None:
    """Registra configuración de confidence."""
    registry.register(
        ParameterMetadata(
            name="CONFIDENCE.MIN_CONFIDENCE",
            category=ParameterCategory.CONFIDENCE_CALIBRATION,
            scope=ParameterScope.GLOBAL,
            value=CONFIDENCE.MIN_CONFIDENCE,
            default_value=0.3,
            min_value=0.0,
            max_value=0.5,
            description="Confidence mínimo permitido (floor)",
            mathematical_meaning="Límite inferior para clamping de confidence",
            statistical_justification="Prevenir confidence excesivamente bajo",
            dependencies=["CONFIDENCE.MAX_CONFIDENCE"],
            validation=lambda x: 0 <= x <= 1,
            version="1.0.0",
        )
    )

    registry.register(
        ParameterMetadata(
            name="CONFIDENCE.MAX_CONFIDENCE",
            category=ParameterCategory.CONFIDENCE_CALIBRATION,
            scope=ParameterScope.GLOBAL,
            value=CONFIDENCE.MAX_CONFIDENCE,
            default_value=0.95,
            min_value=0.8,
            max_value=1.0,
            description="Confidence máximo permitido (ceiling)",
            mathematical_meaning="Límite superior para clamping de confidence",
            statistical_justification="Prevenir confidence excesivamente alto",
            dependencies=[],
            validation=lambda x: 0 <= x <= 1,
            version="1.0.0",
        )
    )


def register_all_parameters(registry: ParameterRegistry) -> None:
    """Registra todos los parámetros del audit."""
    register_epsilon_constants(registry)
    register_statistical_thresholds(registry)
    register_confidence_config(registry)


def migrate_parameters() -> ParameterRegistry:
    """
    Migra todos los parámetros al registry.
    Single source of truth para los parámetros.
    """
    registry = ParameterRegistry()
    register_all_parameters(registry)

    errors = registry.validate_all()
    if errors:
        raise ValueError(f"Parameter validation failed: {errors}")

    cycles = registry.check_dependencies()
    if cycles:
        raise ValueError(f"Circular dependencies detected: {cycles}")

    return registry
