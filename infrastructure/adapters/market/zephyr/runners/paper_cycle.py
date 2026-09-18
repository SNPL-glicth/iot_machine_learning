"""Funciones puras de un ciclo de PaperBotRunner.

Separadas de la clase orquestadora para mantener la gobernanza de ≤180 líneas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from iot_machine_learning.domain.entities.market.prediction.lifecycle import (
    PredictionStatus,
)
from iot_machine_learning.domain.entities.market.prediction.resolver import (
    OutcomeResolver,
)
from iot_machine_learning.domain.entities.market.calibration.gate import EvidenceRecord
from iot_machine_learning.infrastructure.adapters.market.zephyr.runners.paper_protocols import (
    row_to_prediction_safe,
)
from iot_machine_learning.infrastructure.adapters.market.zephyr.runners.paper_report import (
    CycleReport,
)

if TYPE_CHECKING:
    from iot_machine_learning.domain.entities.market.calibration.gate import EvidenceGate
    from iot_machine_learning.domain.entities.market.costs import CostModel
    from iot_machine_learning.infrastructure.adapters.market.zephyr.runners.paper_protocols import (
        EvidenceRepoProtocol,
        FeedProtocol,
        PredictionRepoProtocol,
    )


def build_evidence_records(
    result,
    signal_ts,
    report: CycleReport,
    *,
    gate: EvidenceGate,
    cost_model: CostModel,
    evidence_log,
    require_positive_net: bool,
    totals,
) -> list[EvidenceRecord]:
    """Clasifica predicciones frescas y construye los registros de evidencia."""
    fresh = [p for p in result.predictions if p.observation.timestamp == signal_ts]
    evidence_by_id = {ev.prediction_id: ev for ev in evidence_log}
    records: list[EvidenceRecord] = []
    for pred in fresh:
        ev = evidence_by_id.get(pred.prediction_id)
        if ev is None:
            continue
        decision = (
            gate.decide_with_costs(ev, expected_gross=pred.expected_return, cost_model=cost_model)
            if require_positive_net
            else gate.decide(ev)
        )
        records.append(EvidenceRecord(evidence=ev, decision=decision))
        totals[decision.action.value] += 1
        report.actions[decision.action] += 1
    if fresh and records:
        last = records[-1]
        report.latest_prob_raw = last.evidence.prob_raw
        report.latest_prob_calibrated = last.evidence.prob_calibrated
    elif fresh:
        report.latest_prob_raw = max(fresh, key=lambda p: p.horizon_seconds).probability_up
    return records


def persist_predictions(
    result,
    signal_ts,
    records: list[EvidenceRecord],
    report: CycleReport,
    *,
    prediction_repo_factory: Callable[[], PredictionRepoProtocol],
    evidence_repo_factory: Callable[[], EvidenceRepoProtocol],
) -> None:
    """Persiste las predicciones y los registros de evidencia en repositorios."""
    persistable = [
        p for p in result.predictions
        if not (p.status is PredictionStatus.INVALIDATED and p.invalidation_reason == "feed_ended")
        or p.observation.timestamp == signal_ts
    ]
    report.predictions_persisted = prediction_repo_factory().save_batch(persistable)
    report.evidence_persisted = evidence_repo_factory().save_batch(records)


def resolve_pending(
    report: CycleReport,
    *,
    symbol: str,
    prediction_repo_factory: Callable[[], PredictionRepoProtocol],
    feed: FeedProtocol,
) -> None:
    """Resuelve predicciones pendientes con el feed actual."""
    prediction_repo = prediction_repo_factory()
    pending = prediction_repo.pending_outcomes(symbol=symbol)
    batch = OutcomeResolver().resolve(
        (row_to_prediction_safe(row) for row in pending), feed
    )
    if batch.resolved:
        prediction_repo.save_batch(list(batch.resolved))
    report.resolved = batch.resolved_count
    report.waiting = batch.waiting_count
