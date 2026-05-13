"""Tests para validar registry de constantes numéricas.

Valida que el registro centralizado de constantes:
1. Mantiene jerarquía correcta de epsilons
2. Tiene thresholds estadísticos coherentes
3. Valida confidence correctamente
4. No tiene inconsistencias cross-module
"""

import pytest

from core.parameters.numerical_constants import EPSILON, STAT_THRESHOLDS, CONFIDENCE
from core.parameters.parameter_validator import ParameterValidator


def test_epsilon_hierarchy():
    """Verifica jerarquía correcta de epsilons.
    
    DIVISION debe ser más pequeño que COMPARISON (más conservador).
    DIVISION debe ser más pequeño que CONFIDENCE.
    """
    assert EPSILON.DIVISION < EPSILON.COMPARISON, (
        f"EPSILON.DIVISION ({EPSILON.DIVISION}) debe ser < "
        f"EPSILON.COMPARISON ({EPSILON.COMPARISON})"
    )
    assert EPSILON.DIVISION < EPSILON.CONFIDENCE, (
        f"EPSILON.DIVISION ({EPSILON.DIVISION}) debe ser < "
        f"EPSILON.CONFIDENCE ({EPSILON.CONFIDENCE})"
    )
    assert EPSILON.COMPARISON > 0, "EPSILON.COMPARISON debe ser > 0"
    assert EPSILON.DIVISION > 0, "EPSILON.DIVISION debe ser > 0"
    assert EPSILON.CONFIDENCE > 0, "EPSILON.CONFIDENCE debe ser > 0"


def test_statistical_thresholds():
    """Verifica thresholds estadísticos."""
    # Z-score thresholds
    assert STAT_THRESHOLDS.Z_SCORE_UPPER > STAT_THRESHOLDS.Z_SCORE_LOWER, (
        f"Z_SCORE_UPPER ({STAT_THRESHOLDS.Z_SCORE_UPPER}) debe ser > "
        f"Z_SCORE_LOWER ({STAT_THRESHOLDS.Z_SCORE_LOWER})"
    )
    assert STAT_THRESHOLDS.Z_SCORE_LOWER == 2.0, "Z_SCORE_LOWER debe ser 2.0"
    assert STAT_THRESHOLDS.Z_SCORE_UPPER == 3.0, "Z_SCORE_UPPER debe ser 3.0"
    
    # Contamination
    assert STAT_THRESHOLDS.CONTAMINATION_DEFAULT > 0, "CONTAMINATION_DEFAULT debe ser > 0"
    assert STAT_THRESHOLDS.CONTAMINATION_DEFAULT < 0.1, (
        f"CONTAMINATION_DEFAULT ({STAT_THRESHOLDS.CONTAMINATION_DEFAULT}) debe ser < 0.1"
    )
    assert STAT_THRESHOLDS.CONTAMINATION_MIN < STAT_THRESHOLDS.CONTAMINATION_DEFAULT, (
        "CONTAMINATION_MIN debe ser < CONTAMINATION_DEFAULT"
    )
    assert STAT_THRESHOLDS.CONTAMINATION_MAX > STAT_THRESHOLDS.CONTAMINATION_DEFAULT, (
        "CONTAMINATION_MAX debe ser > CONTAMINATION_DEFAULT"
    )
    
    # IQR
    assert STAT_THRESHOLDS.IQR_FENCE_MULTIPLIER == 1.5, "IQR_FENCE_MULTIPLIER debe ser 1.5"
    
    # LOF
    assert STAT_THRESHOLDS.LOF_MAX_NEIGHBORS == 20, "LOF_MAX_NEIGHBORS debe ser 20"


def test_confidence_validation():
    """Verifica validación de confidence."""
    # Floor
    assert CONFIDENCE.validate(0.1) == CONFIDENCE.MIN_CONFIDENCE, (
        f"validate(0.1) debe retornar MIN_CONFIDENCE ({CONFIDENCE.MIN_CONFIDENCE})"
    )
    assert CONFIDENCE.validate(0.0) == CONFIDENCE.MIN_CONFIDENCE
    
    # Ceiling
    assert CONFIDENCE.validate(0.99) == CONFIDENCE.MAX_CONFIDENCE, (
        f"validate(0.99) debe retornar MAX_CONFIDENCE ({CONFIDENCE.MAX_CONFIDENCE})"
    )
    assert CONFIDENCE.validate(1.0) == CONFIDENCE.MAX_CONFIDENCE
    
    # Normal
    assert CONFIDENCE.validate(0.5) == 0.5, "validate(0.5) debe retornar 0.5"
    assert CONFIDENCE.validate(CONFIDENCE.MIN_CONFIDENCE) == CONFIDENCE.MIN_CONFIDENCE
    assert CONFIDENCE.validate(CONFIDENCE.MAX_CONFIDENCE) == CONFIDENCE.MAX_CONFIDENCE


def test_confidence_range():
    """Verifica rango de confidence."""
    assert CONFIDENCE.MIN_CONFIDENCE == 0.3, "MIN_CONFIDENCE debe ser 0.3"
    assert CONFIDENCE.MAX_CONFIDENCE == 0.95, "MAX_CONFIDENCE debe ser 0.95"
    assert CONFIDENCE.MIN_CONFIDENCE < CONFIDENCE.MAX_CONFIDENCE, (
        "MIN_CONFIDENCE debe ser < MAX_CONFIDENCE"
    )


def test_parameter_consistency():
    """Verifica consistencia cross-module."""
    from core.parameters.parameter_migration import register_all_parameters
    from core.parameters.parameter_registry import ParameterRegistry
    
    registry = ParameterRegistry()
    register_all_parameters(registry)
    validator = ParameterValidator(registry=registry)
    issues = validator.validate_all()
    
    if issues:
        pytest.fail(f"Inconsistencias detectadas:\n" + "\n".join(issues))


def test_parameter_validator_epsilon_hierarchy():
    """Verifica método específico de validación de jerarquía."""
    from core.parameters.parameter_migration import register_all_parameters
    from core.parameters.parameter_registry import ParameterRegistry
    
    # Skip if parameters already registered (registry is singleton)
    try:
        registry = ParameterRegistry()
        register_all_parameters(registry)
    except ValueError as e:
        if "already registered" in str(e):
            pytest.skip("Parameters already registered by previous test")
        raise
    
    validator = ParameterValidator(registry=registry)
    assert validator.validate_epsilon_hierarchy(), "validate_epsilon_hierarchy debe retornar True"


def test_frozen_immutability():
    """Verifica que configs sean inmutables (frozen dataclasses)."""
    # Intentar modificar debería fallar
    with pytest.raises(dataclasses.FrozenInstanceError):
        EPSILON.COMPARISON = 1.0
    
    with pytest.raises(dataclasses.FrozenInstanceError):
        STAT_THRESHOLDS.Z_SCORE_LOWER = 1.5
    
    with pytest.raises(dataclasses.FrozenInstanceError):
        CONFIDENCE.MIN_CONFIDENCE = 0.2


def test_singleton_instances():
    """Verifica que singletons sean instancias únicas."""
    from core.parameters.numerical_constants import EPSILON as eps1, STAT_THRESHOLDS as stat1
    from core.parameters.numerical_constants import EPSILON as eps2, STAT_THRESHOLDS as stat2
    
    assert eps1 is eps2, "EPSILON debe ser singleton (misma instancia)"
    assert stat1 is stat2, "STAT_THRESHOLDS debe ser singleton (misma instancia)"


# Import dataclasses para test de inmutabilidad
import dataclasses
