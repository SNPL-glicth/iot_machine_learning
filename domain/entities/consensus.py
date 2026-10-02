"""Entidades puras de consenso por sincronización topológica de Kuramoto.

Puro: solo biblioteca estándar (dataclasses, typing, math).
Sin dependencias de infraestructura ni librerías de terceros (numpy, scipy, torch).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .representation_evidence import SystemOperationalState


@dataclass(frozen=True)
class KuramotoGateConfig:
    """Configuración analítica y desacoplada de la compuerta de Kuramoto."""

    delta_t: float = 0.05
    alarm_phase: float = math.pi / 2.0
    refractory_steps: int = 2
    coupling_gain_ratio: float = 0.75  # Acoplamiento mutuo subcrítico K = 0.75 * K_c en reposo
    forcing_gain_ratio: float = 3.5  # Fuerza de atracción de alarma = ratio * K * E_i
    burst_velocity_threshold: float = 0.30
    budget_penalty_weight: float = 0.1
    min_shock_threshold: float = 0.35
    evidence_floor: float = 0.35  # Filtro mínimo de evidencia activa requerida para disparo
    expert_frequencies: Mapping[str, float] | None = None
    base_state_thresholds: Mapping[SystemOperationalState, float] | None = None


@dataclass(frozen=True)
class KuramotoState:
    """Estado macroscópico y fases locales del ensamble de osciladores en el paso t."""

    step: int
    order_parameter: float  # r in [0, 1]
    global_phase: float  # psi in [-pi, pi]
    phase_velocity: float  # dr / dt
    phases: Mapping[str, float]  # theta_i in [0, 2*pi)


@dataclass(frozen=True)
class ConsensusDecision:
    """Decisión inmutable emitida por la compuerta de consenso de Kuramoto."""

    step: int
    operational_state: SystemOperationalState
    order_parameter: float  # r(t) in [0, 1]
    phase_velocity: float  # dr / dt
    dynamic_threshold: float  # r_c(t)
    is_triggered: bool  # (r >= r_c) or (dr/dt >= burst and r >= min_shock)
    reason: str
    active_expert_weights: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, float] | None = None

    @property
    def martingale_value(self) -> float:
        """Compatibilidad duck-typing con CausalAggregator (escala [0, 100])."""
        return round(self.order_parameter * 100.0, 4)
