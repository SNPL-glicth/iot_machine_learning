"""Registry centralizado de parámetros del sistema.

Single source of truth para todos los parámetros.
Principio: Single Responsibility - solo gestiona parámetros.
"""

from __future__ import annotations

import time
from abc import ABC
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class ParameterCategory(Enum):
    """Categorías de parámetros según audit."""
    MATHEMATICAL_CONSTANT = "mathematical_constant"
    NUMERICAL_STABILITY = "numerical_stability"
    STATISTICAL_THRESHOLD = "statistical_threshold"
    HEURISTIC_THRESHOLD = "heuristic_threshold"
    PERFORMANCE_TUNING = "performance_tuning"
    INFRASTRUCTURE_POLICY = "infrastructure_policy"
    SECURITY_POLICY = "security_policy"
    EXPERIMENTAL = "experimental"
    ADAPTIVE_CONTROL = "adaptive_control"
    CONFIDENCE_CALIBRATION = "confidence_calibration"


class ParameterScope(Enum):
    """Scope de aplicación del parámetro."""
    GLOBAL = "global"
    PER_ENGINE = "per_engine"
    PER_SERIES = "per_series"
    PER_REGIME = "per_regime"
    DYNAMIC = "dynamic"
    STATISTICAL = "statistical"


@dataclass
class ParameterMetadata:
    """Metadata completa de un parámetro."""
    name: str
    category: ParameterCategory
    scope: ParameterScope
    value: Any
    default_value: Any
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    description: str = ""
    mathematical_meaning: str = ""
    statistical_justification: str = ""
    dependencies: List[str] = field(default_factory=list)
    validation: Optional[Callable[[Any], bool]] = None
    version: str = "1.0.0"
    last_modified: str = ""
    modified_by: str = "system"

    def validate_value(self) -> bool:
        """Valida el valor actual."""
        if self.min_value is not None and self.value < self.min_value:
            return False
        if self.max_value is not None and self.value > self.max_value:
            return False

        if self.validation and not self.validation(self.value):
            return False

        return True


@dataclass
class ParameterChange:
    """Registro de un cambio de parámetro."""
    parameter_name: str
    old_value: Any
    new_value: Any
    reason: str
    changed_by: str
    timestamp: float
    version_before: str
    version_after: str


class ParameterRegistry:
    """
    Registry centralizado de parámetros.

    Responsabilidades:
    - Registrar parámetros con metadata
    - Validar valores y dependencies
    - Mantener audit trail de cambios
    - Version tracking
    """

    _instance: Optional[ParameterRegistry] = None

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "_initialized"):
            self._parameters: Dict[str, ParameterMetadata] = {}
            self._history: List[ParameterChange] = []
            self._initialized = True

    def reset(self) -> None:
        """Reset registry state (for testing)."""
        self._parameters.clear()
        self._history.clear()

    def register(self, metadata: ParameterMetadata) -> None:
        """Registra un parámetro."""
        if metadata.name in self._parameters:
            raise ValueError(f"Parameter {metadata.name} already registered")

        if not metadata.validate_value():
            raise ValueError(f"Invalid value for {metadata.name}")

        self._parameters[metadata.name] = metadata

    def get(self, name: str) -> Optional[ParameterMetadata]:
        """Obtiene metadata de un parámetro."""
        return self._parameters.get(name)

    def get_value(self, name: str) -> Any:
        """Obtiene valor de un parámetro."""
        meta = self.get(name)
        if meta is None:
            raise KeyError(f"Parameter {name} not found")
        return meta.value

    def set_value(
        self, name: str, new_value: Any, reason: str, changed_by: str = "system"
    ) -> None:
        """Actualiza valor de un parámetro con audit trail."""
        meta = self.get(name)
        if meta is None:
            raise KeyError(f"Parameter {name} not found")

        old_value = meta.value
        old_version = meta.version

        meta.value = new_value
        if not meta.validate_value():
            meta.value = old_value
            raise ValueError(f"Invalid value {new_value} for {name}")

        version_parts = meta.version.split(".")
        version_parts[-1] = str(int(version_parts[-1]) + 1)
        new_version = ".".join(version_parts)
        meta.version = new_version
        meta.last_modified = time.strftime("%Y-%m-%d %H:%M:%S")
        meta.modified_by = changed_by

        change = ParameterChange(
            parameter_name=name,
            old_value=old_value,
            new_value=new_value,
            reason=reason,
            changed_by=changed_by,
            timestamp=time.time(),
            version_before=old_version,
            version_after=new_version,
        )
        self._history.append(change)

    def validate_all(self) -> List[str]:
        """Valida todos los parámetros."""
        errors = []
        for name, meta in self._parameters.items():
            if not meta.validate_value():
                errors.append(f"Validation failed for {name}: {meta.value}")
        return errors

    def check_dependencies(self) -> List[str]:
        """Detecta circular dependencies."""
        def has_cycle(param: str, visited: set, stack: set) -> Optional[List[str]]:
            visited.add(param)
            stack.add(param)

            meta = self._parameters.get(param)
            if meta and meta.dependencies:
                for dep in meta.dependencies:
                    if dep not in visited:
                        cycle = has_cycle(dep, visited, stack)
                        if cycle:
                            return cycle
                    elif dep in stack:
                        return [param, dep]

            stack.remove(param)
            return None

        cycles = []
        visited = set()

        for param in self._parameters:
            if param not in visited:
                cycle = has_cycle(param, visited, set())
                if cycle:
                    cycles.append(" -> ".join(cycle))

        return cycles

    def get_by_category(self, category: ParameterCategory) -> List[ParameterMetadata]:
        """Obtiene parámetros por categoría."""
        return [m for m in self._parameters.values() if m.category == category]

    def get_by_scope(self, scope: ParameterScope) -> List[ParameterMetadata]:
        """Obtiene parámetros por scope."""
        return [m for m in self._parameters.values() if m.scope == scope]

    def get_history(
        self, parameter_name: Optional[str] = None, limit: int = 100
    ) -> List[ParameterChange]:
        """Obtiene historial de cambios."""
        if parameter_name:
            history = [c for c in self._history if c.parameter_name == parameter_name]
        else:
            history = self._history
        return history[-limit:]

    def export_config(self) -> dict:
        """Exporta configuración completa."""
        return {
            name: {
                "value": meta.value,
                "version": meta.version,
                "last_modified": meta.last_modified,
            }
            for name, meta in self._parameters.items()
        }

    def get_summary(self) -> dict:
        """Resumen del registry."""
        return {
            "total_parameters": len(self._parameters),
            "by_category": {
                cat.value: len(self.get_by_category(cat)) for cat in ParameterCategory
            },
            "by_scope": {
                scope.value: len(self.get_by_scope(scope)) for scope in ParameterScope
            },
            "total_changes": len(self._history),
        }
