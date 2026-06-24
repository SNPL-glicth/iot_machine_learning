"""PipelineExecutorFactory — IMP-3 + PIPE-2.

Replaces the former module-level ``_pipeline_executor`` singleton.
Each call to :meth:`create` returns a **fresh** :class:`PipelineExecutor`
with its own ``_phases`` list, so two concurrent pipeline runs cannot
share phase-local mutable state (e.g. SanitizePhase resolving stores
from ``ctx.orchestrator``, FusePhase collecting Hampel flags, etc.).

Instance cost is ~2 KB (13 stateless phase objects + one
``AssemblyPhase``) — acceptable for per-request allocation.

``flags_snapshot`` is accepted today but does **not** yet influence
phase composition: every request instantiates the same 13 phases in
the same order. The argument is plumbed through so future flag-driven
phase selection (skipping heavy phases like ``NarrativeUnificationPhase``
based on a feature flag) has a clean hook without another refactor.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .pipeline_executor import PipelineExecutor
from ..compliance import ComplianceExporter
from ..sanitize import SanitizePhase
from .phases.action_guard_phase import ActionGuardPhase
from .phases.boundary_check_phase import BoundaryCheckPhase
from .phases.coherence_check_phase import CoherenceCheckPhase
from .phases.confidence_calibration_phase import ConfidenceCalibrationPhase
from .phases.decision_arbiter_phase import DecisionArbiterPhase
from .phases.explain_phase import ExplainPhase
from .phases.narrative_unification_phase import NarrativeUnificationPhase
from .phases.perceive_phase import PerceivePhase
from .phases.predict_phase import PredictPhase
from .phases.prediction_readiness_gate import PredictionReadinessGate
from .phases.adapt_phase import AdaptPhase
from .phases.inhibit_phase import InhibitPhase
from .phases.fuse_phase import FusePhase

# PIPE-3: Fases opcionales instanciadas solo si flag activo.
# Nota: AdaptPhase, FusePhase e InhibitPhase tienen estado
# compartido a nivel módulo — NO son candidatas a lazy init.
from ml_service.config.feature_flags import get_feature_flags

logger = logging.getLogger(__name__)


class PipelineExecutorFactory:
    """Produces a fresh :class:`PipelineExecutor` per request.

    Callers are expected to invoke :meth:`create` once per pipeline run
    and then discard the returned instance. No caching is performed.
    """

    def __init__(self, compliance_exporter: Optional[ComplianceExporter] = None) -> None:
        self._compliance_exporter = compliance_exporter

    def create(self, flags_snapshot: Optional[Any] = None) -> PipelineExecutor:
        """Return a new :class:`PipelineExecutor`.

        Args:
            flags_snapshot: Feature-flag snapshot for this run. If None,
                flags are read fresh from the environment.
        """
        flags = flags_snapshot or get_feature_flags()

        phases = [
            SanitizePhase(),
            BoundaryCheckPhase(),
            PredictionReadinessGate(),
            PerceivePhase(),
            PredictPhase(),
            FusePhase(),
            InhibitPhase(),
            AdaptPhase(),
        ]

        # Fases opcionales — instanciar solo si flag activo (PIPE-3)
        if flags.ML_DECISION_ARBITER_ENABLED:
            phases.append(DecisionArbiterPhase())
        if flags.ML_COHERENCE_CHECK_ENABLED:
            phases.append(CoherenceCheckPhase())
        if flags.ML_CONFIDENCE_CALIBRATION_ENABLED:
            phases.append(ConfidenceCalibrationPhase())
        if flags.ML_ACTION_GUARD_ENABLED:
            phases.append(ActionGuardPhase())
        if flags.ML_EXPLAINABILITY_ENABLED:
            phases.append(ExplainPhase())
        if flags.ML_NARRATIVE_ENABLED:
            phases.append(NarrativeUnificationPhase())

        return PipelineExecutor(
            phases=phases,
            compliance_exporter=self._compliance_exporter,
        )


__all__ = ["PipelineExecutorFactory"]
