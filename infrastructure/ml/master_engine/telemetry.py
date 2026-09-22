"""Telemetry and decision trace utilities for Master Engine (ISO 22989)."""

from __future__ import annotations

import hashlib
import struct
from typing import Any
import numpy as np

from domain.entities.rosa_roja.execution import ExecutionPlan


def compute_telemetry_hash(delta_state: np.ndarray, delta_time: float) -> str:
    """Compute deterministic SHA256 telemetry hash for ISO 22989 traceability."""
    try:
        data = delta_state.tobytes() + struct.pack("<d", float(delta_time))
        return hashlib.sha256(data).hexdigest()[:16]
    except Exception:
        return "telemetry_err_hash"


def extract_base_trace(plan: ExecutionPlan) -> dict[str, Any]:
    """Extract existing decision trace from execution plan envelope or veto details."""
    if plan.envelope and plan.envelope.metadata and "decision_trace" in plan.envelope.metadata:
        return dict(plan.envelope.metadata["decision_trace"])
    if getattr(plan, "veto_details", None) and isinstance(plan.veto_details, dict):
        if "decision_trace" in plan.veto_details:
            return dict(plan.veto_details["decision_trace"])
    return {}


def assemble_master_trace(
    base_trace: dict[str, Any],
    phi_moe_base: float,
    i_cvar: float,
    lambda_t_crono: float,
    certeza: float,
    magnitud_objetivo: float,
    momentum_veto: float,
    shadow_mode: bool,
    risk_verdict: dict[str, Any],
    temporal_verdict: dict[str, Any],
    telemetry_hash: str,
) -> dict[str, Any]:
    """Assemble complete ISO 22989 decision trace dictionary."""
    return {
        **base_trace,
        "telemetry_hash": telemetry_hash,
        "phi_moe_base": float(phi_moe_base),
        "I_cvar": float(i_cvar),
        "lambda_t_crono": float(lambda_t_crono),
        "certeza": float(certeza),
        "phi_redrose": float(certeza),
        "magnitud_objetivo": float(magnitud_objetivo),
        "momentum_veto": float(momentum_veto),
        "governing_component": "phi_moe_base" if shadow_mode else "master_equation",
        "execution_mode": "shadow" if shadow_mode else "active",
        "risk_engine_shadow": risk_verdict,
        "temporal_engine_shadow": temporal_verdict,
    }
