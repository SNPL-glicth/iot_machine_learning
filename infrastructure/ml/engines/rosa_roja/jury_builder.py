"""Jury builder: configura los miembros del jurado MoE para Rosa Roja."""

from __future__ import annotations

import logging
from typing import List, Sequence

from domain.ports.rosa_roja.expert_jury import ExpertJuryPort

logger = logging.getLogger(__name__)


def build_default_moe_jury(
    custom_jury: Sequence[ExpertJuryPort] | None = None,
) -> List[ExpertJuryPort]:
    """Construye jurado de expertos MoE por defecto para Rosa Roja.

    Si se proporciona custom_jury, se utiliza directamente. De lo contrario,
    se ensamblan Taylor, Kalman y Statistical como evaluadores de trayectoria.
    """
    if custom_jury is not None:
        return list(custom_jury)

    from iot_machine_learning.infrastructure.ml.adapters import (
        KalmanExpertAdapter,
        StatisticalExpertAdapter,
        TaylorExpertAdapter,
    )
    from iot_machine_learning.infrastructure.ml.engines.kalman.engine import (
        KalmanPredictionEngine,
    )
    from iot_machine_learning.infrastructure.ml.engines.statistical import (
        StatisticalPredictionEngine,
    )
    from iot_machine_learning.infrastructure.ml.engines.taylor.engine import (
        TaylorPredictionEngine,
    )

    jury: List[ExpertJuryPort] = []
    try:
        taylor_engine = TaylorPredictionEngine()
        jury.append(TaylorExpertAdapter(engine=taylor_engine))
    except Exception as exc:
        logger.warning("rosa_roja_jury_taylor_failed", extra={"error": str(exc)})

    try:
        kalman_engine = KalmanPredictionEngine()
        jury.append(KalmanExpertAdapter(engine=kalman_engine))
    except Exception as exc:
        logger.warning("rosa_roja_jury_kalman_failed", extra={"error": str(exc)})

    try:
        stat_engine = StatisticalPredictionEngine()
        jury.append(StatisticalExpertAdapter(engine=stat_engine))
    except Exception as exc:
        logger.warning("rosa_roja_jury_stat_failed", extra={"error": str(exc)})

    return jury
