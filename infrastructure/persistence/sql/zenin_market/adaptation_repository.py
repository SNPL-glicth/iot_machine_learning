"""AdaptationRepository (FASE 8) — propuestas y versiones del modelo.

Append-only por diseño: las propuestas (aceptadas O rechazadas) y las
versiones jamás se modifican ni se borran. Crear una versión desactiva
la activa anterior en la misma transacción (parent_version_id + is_active
preservan la cadena completa: v1 → v2 → ...).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Connection

from iot_machine_learning.domain.entities.market.adaptation.guard import (
    GuardCheck,
    GuardResult,
)
from iot_machine_learning.domain.entities.market.adaptation.proposer import (
    WeightProposal,
)

__all__ = ["AdaptationRepository"]


def proposal_to_row(
    proposal: WeightProposal,
    *,
    proposal_id: str,
    status: str,
    version_id: int | None = None,
    rejected_reason: str | None = None,
    checks: tuple[GuardCheck, ...] | None = None,
) -> dict[str, Any]:
    """Mapeo puro propuesta -> fila adaptation_proposals."""
    return {
        "proposal_id": proposal_id,
        "version_id": version_id,
        "expert": proposal.expert,
        "regime": proposal.regime,
        "horizon_seconds": proposal.horizon_seconds,
        "current_weight": proposal.current_weight,
        "proposed_weight": proposal.proposed_weight,
        "observed_reward": proposal.observed_reward,
        "calibration": proposal.calibration,
        "sample_size": proposal.sample_size,
        "reason": proposal.reason,
        "status": status,
        "rejected_reason": rejected_reason,
        "guard_checks": (json.dumps([_check_to_json(c) for c in checks]) if checks else None),
        "parent_version_id": (int(proposal.parent_version) if proposal.parent_version else None),
        "created_at": proposal.created_at,
    }


def row_to_proposal(row: dict[str, Any]) -> WeightProposal:
    """Reconstruye la propuesta desde su fila (para auditoría)."""
    return WeightProposal(
        expert=row["expert"],
        regime=row["regime"],
        horizon_seconds=row["horizon_seconds"],
        current_weight=row["current_weight"],
        proposed_weight=row["proposed_weight"],
        observed_reward=row["observed_reward"] or 0.0,
        calibration=row["calibration"] or 0.0,
        sample_size=row["sample_size"],
        accuracy=0.0,  # la fila no persiste accuracy: la razón textual la conserva
        reason=row["reason"],
        created_at=row["created_at"],
        parent_version=(
            str(row["parent_version_id"]) if row.get("parent_version_id") is not None else None
        ),
    )


def _check_to_json(check: GuardCheck) -> dict[str, Any]:
    return {"name": check.name, "ok": check.ok, "detail": check.detail}


class AdaptationRepository:
    """Registro de propuestas y versiones (audit trail de FASE 8)."""

    def __init__(self, conn: Connection) -> None:
        self._conn = conn

    # ─── propuestas ────────────────────────────────────────────────────────

    def record_proposal(
        self,
        proposal: WeightProposal,
        guard: GuardResult,
        *,
        proposal_id: str | None = None,
        version_id: int | None = None,
    ) -> str:
        """Guarda la propuesta con su veredicto (ACCEPTED/REJECTED).

        Siempre se guarda: el sistema audita TODO lo que consideró cambiar.
        """
        pid = proposal_id or f"prop-{uuid.uuid4().hex[:12]}"
        status = "accepted" if guard.passed else "rejected"
        rejected_reason = None
        if not guard.passed:
            failed = guard.failed_checks
            rejected_reason = "; ".join(c.name for c in failed) or "guardrails"
        row = proposal_to_row(
            proposal,
            proposal_id=pid,
            status=status,
            version_id=version_id,
            rejected_reason=rejected_reason,
            checks=guard.checks,
        )
        self._conn.execute(
            text(
                """
                INSERT INTO adaptation_proposals (
                    proposal_id, version_id, expert, regime, horizon_seconds,
                    current_weight, proposed_weight, observed_reward,
                    calibration, sample_size, reason, status, rejected_reason,
                    guard_checks, parent_version_id, created_at
                ) VALUES (
                    :proposal_id, :version_id, :expert, :regime, :horizon_seconds,
                    :current_weight, :proposed_weight, :observed_reward,
                    :calibration, :sample_size, :reason, :status, :rejected_reason,
                    :guard_checks, :parent_version_id, :created_at
                )
                """
            ),
            row,
        )
        return pid

    def proposal_history(
        self,
        *,
        expert: str | None = None,
        status: str | None = None,
        proposal_id_prefix: str | None = None,
        limit: int = 50,
    ) -> tuple[dict[str, Any], ...]:
        """Audit trail de propuestas (filtrable por experto/estado/prefix)."""
        sql = "SELECT * FROM adaptation_proposals"
        clauses: list[str] = []
        params: dict[str, Any] = {}
        if expert is not None:
            clauses.append("expert = :expert")
            params["expert"] = expert
        if status is not None:
            clauses.append("status = :status")
            params["status"] = status
        if proposal_id_prefix is not None:
            clauses.append("proposal_id LIKE :prefix")
            params["prefix"] = f"{proposal_id_prefix}%"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT :limit"
        params["limit"] = limit
        rows = self._conn.execute(text(sql), params).mappings().all()
        return tuple(dict(row) for row in rows)

    # ─── versiones ─────────────────────────────────────────────────────────

    def latest_version(self) -> dict[str, Any] | None:
        """Versión activa del modelo (la más reciente creada)."""
        row = (
            self._conn.execute(
                text(
                    "SELECT * FROM model_versions "
                    "WHERE is_active = 1 ORDER BY version_id DESC LIMIT 1"
                )
            )
            .mappings()
            .first()
        )
        return dict(row) if row else None

    def create_version(
        self,
        *,
        weights: dict[str, dict[str, float]],
        calibration: dict[str, float],
        reason: str,
        proposal_ids: list[str] | None = None,
        guard_checks: list[dict[str, Any]] | None = None,
        parent_version_id: int | None = None,
        created_at: float | None = None,
    ) -> int:
        """Crea la versión nueva y desactiva la activa (append-only)."""
        created_at = created_at or time.time()
        if parent_version_id is None:
            current = self.latest_version()
            parent_version_id = int(current["version_id"]) if current else None
        self._conn.execute(text("UPDATE model_versions SET is_active = 0"))
        result = self._conn.execute(
            text(
                """
                INSERT INTO model_versions (
                    created_at, weights, calibration, reason,
                    parent_version_id, proposal_id, guard_checks, is_active
                ) VALUES (
                    :created_at, :weights, :calibration, :reason,
                    :parent_version_id, :proposal_id, :guard_checks, 1
                )
                """
            ),
            {
                "created_at": created_at,
                "weights": json.dumps(weights),
                "calibration": json.dumps(calibration),
                "reason": reason,
                "parent_version_id": parent_version_id,
                "proposal_id": (",".join(proposal_ids) if proposal_ids else None),
                "guard_checks": (json.dumps(guard_checks) if guard_checks else None),
            },
        )
        return int(result.lastrowid)

    def list_versions(self, limit: int = 20) -> tuple[dict[str, Any], ...]:
        """Cadena completa de versiones, de la más reciente a la antigua."""
        rows = (
            self._conn.execute(
                text("SELECT * FROM model_versions ORDER BY version_id DESC LIMIT :limit"),
                {"limit": limit},
            )
            .mappings()
            .all()
        )
        return tuple(dict(row) for row in rows)

    def version(self, version_id: int) -> dict[str, Any] | None:
        row = (
            self._conn.execute(
                text("SELECT * FROM model_versions WHERE version_id = :version_id"),
                {"version_id": version_id},
            )
            .mappings()
            .first()
        )
        return dict(row) if row else None
