"""Telemetry and decision trace utilities for Master Engine (ISO/IEC 22989)."""

from __future__ import annotations

import hashlib
import struct
from typing import Any
import numpy as np

from domain.entities.rosa_roja.execution import ExecutionPlan
from domain.entities.rosa_roja.decision_trace import (
    DecisionTraceRecord,
    SubsystemStatus,
)


def compute_telemetry_hash(delta_state: np.ndarray, delta_time: float) -> str:
    """Compute deterministic SHA256 telemetry hash for ISO 22989 traceability."""
    try:
        arr = np.asarray(delta_state, dtype=np.float64).flatten()
        data = arr.tobytes() + struct.pack("<d", float(delta_time))
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
    manifold_audit: Any | None = None,
    variable_destino: float | None = None,
) -> dict[str, Any]:
    """Assemble complete ISO 22989 decision trace dictionary using DecisionTraceRecord."""
    health: dict[str, SubsystemStatus] = {
        "moe_gating": SubsystemStatus.HEALTHY if phi_moe_base > 0 else SubsystemStatus.DEGRADED,
        "risk_engine": SubsystemStatus.HEALTHY if risk_verdict else SubsystemStatus.DEGRADED,
        "temporal_engine": SubsystemStatus.HEALTHY if temporal_verdict else SubsystemStatus.DEGRADED,
        "manifold_engine": SubsystemStatus.SHADOW if manifold_audit is not None else SubsystemStatus.DEGRADED,
    }
    if risk_verdict and not risk_verdict.get("veto_riesgo", 1):
        health["risk_engine"] = SubsystemStatus.VETOED

    diag: dict[str, Any] = {
        "risk_engine_shadow": risk_verdict,
        "temporal_engine_shadow": temporal_verdict,
    }
    if manifold_audit is not None:
        try:
            diag["manifold_divergence"] = float(manifold_audit.divergence)
            diag["manifold_volume_rate"] = float(manifold_audit.volume_rate)
            diag["manifold_regime"] = str(manifold_audit.stability_verdict)
            diag["is_4d_projected"] = bool(manifold_audit.is_4d_projected)
            if hasattr(manifold_audit, "to_telemetry_trace"):
                diag["manifold_telemetry"] = manifold_audit.to_telemetry_trace()
        except Exception as exc:
            health["manifold_engine"] = SubsystemStatus.FAILED
            diag["manifold_error"] = str(exc)

    record = DecisionTraceRecord.create(
        telemetry_hash=telemetry_hash,
        phi_moe_base=phi_moe_base,
        i_cvar=i_cvar,
        lambda_t_crono=lambda_t_crono,
        certeza=certeza,
        magnitud_objetivo=magnitud_objetivo,
        momentum_veto=momentum_veto,
        shadow_mode=shadow_mode,
        action_verdict="EXECUTE" if (certeza >= 0.5 and momentum_veto > 0.0) else "HOLD",
        state_dim=int(base_trace.get("state_dimension", 1)),
        mahalanobis_d=float(base_trace.get("mahalanobis_dist", base_trace.get("mahal_dist", 0.0))),
        dynamic_threshold=float(base_trace.get("noise_threshold", 3.0)),
        is_outlier=bool(base_trace.get("is_outlier", False)),
        variable_destino=variable_destino,
        subsystem_health=health,
        diagnostics=diag,
    )
    trace = record.to_dict()
    for k, v in base_trace.items():
        if k not in trace:
            trace[k] = v
    return trace
