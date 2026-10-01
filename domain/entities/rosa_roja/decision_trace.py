"""ISO/IEC 22989 & ISO/IEC 25010 Decision Trace Record (Domain Layer).

Provides formal governance, mathematical transparency, and fault observability
for single inference steps across all ML subsystems without breaking legacy contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import struct
import time
from typing import Any, Mapping
import numpy as np


class ExecutionMode(str, Enum):
    """Operational mode under ISO/IEC 22989 §5.17 AI Lifecycle & Governance."""
    ACTIVE = "active"
    SHADOW = "shadow"


class SubsystemStatus(str, Enum):
    """Reliability state under ISO/IEC 25010 Fault Tolerance & Observability."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    VETOED = "vetoed"
    FAILED = "failed"
    SHADOW = "shadow"


@dataclass(frozen=True, slots=True)
class DecisionTraceRecord:
    """Formal audit record fulfilling ISO/IEC 22989 §5.3 and ISO/IEC 25010."""

    timestamp: float
    telemetry_hash: str
    execution_mode: ExecutionMode
    governing_component: str
    action_verdict: str

    # Governing Decision Terms (drive ExecutionPlan)
    certeza: float
    magnitud_objetivo: float
    momentum_veto: float
    gamma_exec: float

    # Subsystem Inputs & Evidence Terms
    phi_moe_base: float
    i_cvar: float
    lambda_t_crono: float

    # Data Quality & Ingestion Context (ISO/IEC 5259)
    state_dim: int
    mahalanobis_d: float
    dynamic_threshold: float
    is_outlier: bool

    # Observational & Research Shadow Diagnostics (Do NOT govern execution)
    variable_destino: float | None = None
    subsystem_health: Mapping[str, SubsystemStatus] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        delta_state: np.ndarray | None = None,
        delta_time: float = 1.0,
        phi_moe_base: float = 0.0,
        i_cvar: float = 1.0,
        lambda_t_crono: float = 1.0,
        certeza: float = 0.0,
        magnitud_objetivo: float = 0.0,
        momentum_veto: float = 1.0,
        gamma_exec: float = 0.5,
        shadow_mode: bool = False,
        action_verdict: str = "HOLD",
        state_dim: int | None = None,
        mahalanobis_d: float = 0.0,
        dynamic_threshold: float = 3.0,
        is_outlier: bool = False,
        variable_destino: float | None = None,
        subsystem_health: Mapping[str, SubsystemStatus] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
        telemetry_hash: str | None = None,
    ) -> DecisionTraceRecord:
        """Constructs and validates a tamper-evident decision trace record."""
        arr = np.asarray(delta_state, dtype=np.float64).flatten() if delta_state is not None else np.array([], dtype=np.float64)
        dim = state_dim if state_dim is not None else (arr.size if arr.size > 0 else 1)
        dt = max(1e-9, float(delta_time))

        if telemetry_hash is not None:
            thash = telemetry_hash
        else:
            try:
                thash = hashlib.sha256(arr.tobytes() + struct.pack("<d", dt)).hexdigest()[:16]
            except Exception:
                thash = "hash_fallback_00"

        mode = ExecutionMode.SHADOW if shadow_mode else ExecutionMode.ACTIVE
        gov = "phi_moe_base" if shadow_mode else "master_equation"

        return cls(
            timestamp=time.time(),
            telemetry_hash=thash,
            execution_mode=mode,
            governing_component=gov,
            action_verdict=action_verdict,
            certeza=float(np.clip(certeza, 0.0, 1.0)),
            magnitud_objetivo=max(0.0, float(magnitud_objetivo)),
            momentum_veto=float(np.clip(momentum_veto, 0.0, 1.0)),
            gamma_exec=float(gamma_exec),
            phi_moe_base=float(np.clip(phi_moe_base, 0.0, 1.0)),
            i_cvar=float(np.clip(i_cvar, 0.0, 1.0)),
            lambda_t_crono=float(np.clip(lambda_t_crono, 0.0, 1.0)),
            state_dim=dim,
            mahalanobis_d=max(0.0, float(mahalanobis_d)),
            dynamic_threshold=max(0.0, float(dynamic_threshold)),
            is_outlier=is_outlier,
            variable_destino=float(variable_destino) if variable_destino is not None else None,
            subsystem_health=dict(subsystem_health) if subsystem_health else {},
            diagnostics=dict(diagnostics) if diagnostics else {},
        )

    def to_dict(self) -> dict[str, Any]:
        """Exports 100% backward-compatible dictionary for ExecutionPlan and audits."""
        trace: dict[str, Any] = {
            "iso_standard": "ISO/IEC 22989:2022 §5.3",
            "telemetry_hash": self.telemetry_hash,
            "timestamp": self.timestamp,
            "execution_mode": self.execution_mode.value,
            "governing_component": self.governing_component,
            "action_verdict": self.action_verdict,
            # Legacy expected keys for tests and adapters
            "phi_moe_base": self.phi_moe_base,
            "I_cvar": self.i_cvar,
            "lambda_t_crono": self.lambda_t_crono,
            "certeza": self.certeza,
            "phi_redrose": self.certeza,
            "magnitud_objetivo": self.magnitud_objetivo,
            "momentum_veto": self.momentum_veto,
            "gamma_exec": self.gamma_exec,
            # ISO 5259 Data Quality context
            "state_dimension": self.state_dim,
            "mahalanobis_d": self.mahalanobis_d,
            "mahalanobis_threshold": self.dynamic_threshold,
            "is_outlier": self.is_outlier,
            "subsystem_health": {
                k: (v.value if isinstance(v, Enum) else str(v))
                for k, v in (self.subsystem_health or {}).items()
            },
        }
        if self.variable_destino is not None:
            trace["variable_destino"] = self.variable_destino
            trace["D_t"] = self.variable_destino
        if self.diagnostics:
            for k, v in self.diagnostics.items():
                if k not in trace:
                    trace[k] = v
        return trace
