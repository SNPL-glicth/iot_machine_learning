"""Entidad de severidad — value object puro."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SeverityResult:
    """Resultado de clasificación de severidad.
    
    Value object puro — sin lógica de negocio.
    Movido desde domain/services/severity_rules.py para romper ciclo de imports.
    
    Args:
        risk_level: Nivel de riesgo ('NONE', 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
        severity: Severidad ('info', 'warning', 'critical')
        action_required: Si requiere acción inmediata
        recommended_action: Acción recomendada (texto)
    """
    
    risk_level: str
    severity: str
    action_required: bool
    recommended_action: str
