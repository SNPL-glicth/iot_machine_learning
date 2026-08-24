"""Integration — circuito FASE 8: propuestas → guardrail → versiones.

Contra MySQL zenin_market real: expert_performance consume SOLO filas
evaluadas; cada propuesta (aceptada o rechazada) queda registrada con su
veredicto; una versión nueva desactiva la activa preservando la cadena
parent_version_id. Usa prefijos TEST- y limpia al final.

Requiere MySQL corriendo; si no, se salta (mismo criterio que el resto).
"""

from __future__ import annotations

import json

import pytest
from iot_machine_learning.domain.entities.market.adaptation import (
    AdaptationGuard,
    WeightProposal,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.adaptation_repository import (
    AdaptationRepository,
    proposal_to_row,
)
from iot_machine_learning.infrastructure.persistence.sql.zenin_market.market_db_connection import (
    ZeninMarketDbConnection,
)
from sqlalchemy import text

pytestmark = pytest.mark.integration

TS0 = 3_000_000.0
SYMBOL = "TEST"


def _proposal(
    *,
    current=0.25,
    proposed=0.31,
    n=183,
    accuracy=0.68,
    reward=0.81,
    parent="1",
    now=TS0,
    expert="momentum",
    regime="TRENDING",
    horizon=900,
) -> WeightProposal:
    return WeightProposal(
        expert=expert,
        regime=regime,
        horizon_seconds=horizon,
        current_weight=current,
        proposed_weight=proposed,
        observed_reward=reward,
        calibration=0.02,
        sample_size=n,
        accuracy=accuracy,
        reason="increase under TRENDING/900s: reward_adjusted +0.81",
        created_at=now,
        parent_version=parent,
    )


@pytest.fixture(scope="module")
def engine():
    if not ZeninMarketDbConnection.health_check():
        pytest.skip("MySQL zenin_market no disponible")
    from iot_machine_learning.infrastructure.persistence.sql.zenin_market.migrations import (
        apply_migrations,
    )

    apply_migrations()
    with ZeninMarketDbConnection.get_connection() as conn:
        # Idempotente ante corridas previas interrumpidas: limpiar TEST-.
        conn.execute(
            text("DELETE FROM adaptation_proposals WHERE proposal_id LIKE 'TEST-%'")
        )
    return ZeninMarketDbConnection.get_engine()


class TestAdaptationCircuit:
    def test_proposal_to_guard_to_version(self, engine) -> None:
        with engine.begin() as conn:
            adaptation = AdaptationRepository(conn)

            # Propuesta sana -> ACCEPT.
            good = _proposal()
            guard = AdaptationGuard(min_n=10, min_history_days=2)
            verdict = guard.evaluate(
                good,
                history_days=3,
                context_weights_after={"momentum": 0.31, "naive": 0.69},
            )
            assert verdict.passed
            pid_ok = adaptation.record_proposal(good, verdict, proposal_id="TEST-prop-ok")
            assert pid_ok == "TEST-prop-ok"

            # Propuesta pobre -> REJECT y aun así se registra.
            bad = _proposal(n=4, accuracy=0.4)
            verdict_bad = guard.evaluate(
                bad,
                history_days=3,
                context_weights_after={"momentum": 0.31, "naive": 0.69},
            )
            assert not verdict_bad.passed
            pid_bad = adaptation.record_proposal(bad, verdict_bad, proposal_id="TEST-prop-bad")
            assert pid_bad == "TEST-prop-bad"

            # El veredicto queda guardado con sus chequeos (audit).
            # El store es compartido con corridas reales: filtrar por TEST-.
            test_rows = [
                r for r in adaptation.proposal_history(proposal_id_prefix="TEST-")
                if r["proposal_id"].startswith("TEST-")
            ]
            by_id = {r["proposal_id"]: r for r in test_rows}
            row = by_id["TEST-prop-ok"]
            checks = json.loads(row["guard_checks"])
            assert isinstance(checks, list) and len(checks) == 9
            assert row["status"] == "accepted"
            row_bad = by_id["TEST-prop-bad"]
            assert "min_n" in row_bad["rejected_reason"]
            assert "statistical" in row_bad["rejected_reason"]

            # Filtro por experto (incluye filas reales: solo verificamos TEST-).
            only = adaptation.proposal_history(expert="momentum", proposal_id_prefix="TEST-")
            assert {"TEST-prop-ok", "TEST-prop-bad"} <= {r["proposal_id"] for r in only}

            # Versión nueva: activa, con la anterior preservada.
            v1 = adaptation.create_version(
                weights={"*|TRENDING|900s": {"momentum": 0.25, "naive": 0.75}},
                calibration={"momentum|TRENDING|900s": {"n": 10}},
                reason="v1 inicial (test)",
                proposal_ids=None,
                created_at=TS0,
            )
            v2 = adaptation.create_version(
                weights={"*|TRENDING|900s": {"momentum": 0.31, "naive": 0.69}},
                calibration={"momentum|TRENDING|900s": {"n": 183}},
                reason="aceptada: momentum gana bajo TRENDING/900s",
                proposal_ids=["TEST-prop-ok"],
                created_at=TS0 + 1,
                parent_version_id=v1,
            )
            assert v2 == v1 + 1
            latest = adaptation.latest_version()
            assert int(latest["version_id"]) == v2
            assert latest["is_active"] == 1
            assert int(latest["parent_version_id"]) == v1
            assert latest["proposal_id"] == "TEST-prop-ok"
            weights = json.loads(latest["weights"])
            assert weights["*|TRENDING|900s"]["momentum"] == pytest.approx(0.31)

            # La cadena completa es consultable (audit trail).
            versions = adaptation.list_versions(limit=10)
            ids = [int(v["version_id"]) for v in versions]
            assert v2 in ids and v1 in ids
            parent = adaptation.version(v2)
            assert int(parent["parent_version_id"]) == v1

            # El mapping puro reconstruye la propuesta desde su fila.
            roundtrip = proposal_to_row(
                good, proposal_id="TEST-prop-ok", status="accepted", checks=verdict.checks
            )
            assert roundtrip["expert"] == "momentum"
            assert roundtrip["proposed_weight"] == pytest.approx(0.31)
            assert roundtrip["status"] == "accepted"

    def test_cleanup(self, engine) -> None:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM adaptation_proposals WHERE proposal_id LIKE 'TEST-%'"))
            conn.execute(
                text(
                    "DELETE FROM model_versions WHERE reason LIKE '%(test)%' "
                    "OR reason LIKE '%aceptada: momentum%'"
                )
            )
            # El test desactivó versiones reales con su create_version:
            # restaurar la activa más reciente que quede.
            conn.execute(
                text(
                    "UPDATE model_versions SET is_active = 1 "
                    "ORDER BY version_id DESC LIMIT 1"
                )
            )
