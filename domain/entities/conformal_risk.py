"""Entidades de riesgo conforme, cotas de martingala y decisiones de compuerta adaptativa.

Puro: solo biblioteca estándar (dataclasses, enum, typing).
Sin dependencias de infraestructura ni librerías de terceros (numpy, scipy, torch).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Mapping

from .representation_evidence import SystemOperationalState


class RiskCertificationStatus(Enum):
    """Estado formal de la evidencia acumulada bajo la cota de Ville."""

    NOMINAL = auto()  # M_t < 1 / alpha (Bajo H0 con probabilidad >= 1 - alpha)
    BORDERLINE = auto()  # M_t en zona de advertencia previa al stopping time
    CERTIFIED_ALARM = auto()  # M_t >= tau_t (Violación certificada de la hipótesis nula)


@dataclass(frozen=True)
class ConformalBound:
    """Cotas de riesgo y parámetros teóricos de calibración online."""

    alpha_target: float  # Nivel de significancia libre de distribución (e.g. 0.01)
    base_threshold: float  # 1.0 / alpha_target (Cota canónica de Ville)
    current_dynamic_threshold: float  # tau_t ajustado por régimen y presupuesto
    stopping_time_guarantee: str = "Ville_Maximal_Inequality_1939"


@dataclass(frozen=True)
class AdaptiveGateDecision:
    """Decisión inmutable emitida por la puerta de decisión multiobjetivo."""

    step: int
    operational_state: SystemOperationalState
    martingale_value: float  # M_t acumulado
    dynamic_threshold: float  # tau_t en el paso t
    certification: RiskCertificationStatus
    is_triggered: bool  # M_t >= dynamic_threshold
    active_expert_weights: Mapping[str, float]  # w_{i, t} derivados del Hedge/OCO
    budget_penalty_factor: float  # Factor multiplicativo por estrés de latencia/cómputo
    reason: str
    metadata: Mapping[str, float] | None = None
